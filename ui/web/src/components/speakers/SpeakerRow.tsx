import { cn } from '../../lib/cn';
import { fmtClock, fmtRationale, genderWord } from '../../lib/format';
import { IconPause, IconPlay, IconRevert } from '../ui/Icons';
import { Tooltip } from '../ui/Tooltip';
import { Select } from '../ui/Select';
import { StagedValue } from '../ui/StagedValue';
import { VoicePicker } from './VoicePicker';
import { SpeakerNameField } from './SpeakerNameField';
import { togglePreview, useAudioPreview } from '../../hooks/useAudioPreview';
import { useStore } from '../../store/useStore';
import type { AgentScriptRow, AgentSpeaker, Speaker, SpeakerTable, VoiceOption } from '../../api/types';
import type { StagedChange } from '../../store/useStore';

/**
 * One speaker.
 *
 * Four things live on this row, and the difference between them matters:
 *
 *   name     EDITABLE, display only. Renaming dirties no stage — see
 *            `SpeakerNameField`.
 *   gender   STAGED. A picker offers male/female; changing it recasts the
 *            voice from the matching-gender pool (`set_gender`).
 *   voice    STAGED. Applied by an explicit button that prices it first.
 *   tone     STAGED, and ABSENT — not disabled — when the engine would
 *            ignore it. `ui/speakers.py::supports_tone` uses the exact
 *            predicate `backend/edits.py::_h_set_tone` warns on, so the
 *            control disappears precisely when the value would be dropped,
 *            and the spec panel states the absence in words so a missing
 *            feature does not read as a bug.
 *
 * Two staging paths reach this row and both have to show. The pickers write
 * `staged` (local, in the browser, not yet posted); the chat box writes the
 * job's `pending.json`, which arrives as `fromAgent`.
 */
export function SpeakerRow({
  speaker,
  table,
  staged,
  fromAgent,
  cues,
  onVoice,
  onTone,
  onGender,
  onRevert,
  editable,
}: {
  speaker: Speaker;
  table: SpeakerTable;
  staged?: StagedChange;
  fromAgent?: AgentSpeaker | null;
  cues?: AgentScriptRow[];
  onVoice: (voice: string) => void;
  onTone: (tone: string) => void;
  onGender: (gender: string) => void;
  onRevert: () => void;
  editable: boolean;
}) {
  const requestSeek = useStore((s) => s.requestSeek);
  const selectedSpeaker = useStore((s) => s.selectedSpeaker);
  const setSelectedSpeaker = useStore((s) => s.setSelectedSpeaker);
  const selected = selectedSpeaker === speaker.id;
  const playing = useAudioPreview();
  const isPlaying = Boolean(speaker.reference_url && playing === speaker.reference_url);
  const dirty = Boolean(staged);
  // The post-MT cast gate. While it is up, each speaker is either auto-detected
  // (gender predicted, voice auto-cast) or chosen by hand; the pickers only
  // apply in the latter mode.
  const awaitingCast = useStore((s) => Boolean(s.progress?.milestones?.awaiting_cast));
  const castMode = useStore((s) => s.castMode[speaker.id] ?? 'autodetect');
  const setCastMode = useStore((s) => s.setCastMode);
  const pickersEnabled = editable && (!awaitingCast || castMode === 'select');

  // The picker's own value is what the CONTROL shows, so it keeps following
  // the local store. The before/after beneath is the separate question of
  // what this row will BECOME, and a server-staged change counts there too.
  const voice = staged?.voice !== undefined ? staged.voice : speaker.voice;
  const tone = staged?.tone !== undefined ? staged.tone : speaker.tone;

  const voiceLabel = (v: string | null | undefined) =>
    (v
      ? (table.voice_options.find((o: VoiceOption) => o.voice === v)?.display
        ?? v.split('-').pop() ?? v)
      : null);
  const toneLabel = (t: string | null | undefined) =>
    (t ? table.tone_options.find((o) => o.value === t)?.label ?? t : null);

  // `staged` holds `fromVoice`/`fromTone` — what the row read as when the
  // picker was first touched — so a second change still compares against the
  // dub rather than against the first change.
  const stagedVoice = staged?.voice !== undefined && staged.voice !== staged.fromVoice
    ? voiceLabel(staged.voice)
    : voiceLabel(fromAgent?.voice.staged);
  const stagedTone = staged?.tone !== undefined && staged.tone !== staged.fromTone
    ? toneLabel(staged.tone)
    : toneLabel(fromAgent?.tone.staged);
  const currentVoice = voiceLabel(
    staged?.fromVoice !== undefined ? staged.fromVoice : speaker.voice);
  const currentTone = toneLabel(
    staged?.fromTone !== undefined ? staged.fromTone : speaker.tone);

  const genderRaw = staged?.gender !== undefined ? staged.gender : speaker.gender;
  const gender = genderWord(genderRaw);
  const stagedGender = staged?.gender !== undefined && staged.gender !== staged.fromGender
    ? genderWord(staged.gender)
    : genderWord(fromAgent?.gender.staged ?? null);
  const currentGender = genderWord(
    staged?.fromGender !== undefined ? staged.fromGender : speaker.gender);
  // Any of the three about to change, from either path.
  const willChange = Boolean(
    (stagedVoice && stagedVoice !== currentVoice)
    || (stagedTone && stagedTone !== currentTone)
    || (stagedGender && stagedGender !== currentGender),
  );

  const rationale = fmtRationale(speaker.voice_rationale);

  return (
    <div
      className={cn(
        'relative flex flex-wrap items-center gap-3 px-3.5 py-2',
        dirty && 'bg-warn/[0.06]',
        selected && 'ring-2 ring-inset ring-brand',
      )}
    >
      {dirty ? (
        <span className="absolute inset-y-0 left-0 w-[3px] bg-warn" aria-hidden="true" />
      ) : null}

      <div className="flex min-w-0 flex-1 basis-[15rem] items-center gap-2.5">
        <button
          type="button"
          onClick={() => setSelectedSpeaker(selected ? null : speaker.id)}
          aria-label={
            selected
              ? `Stop aiming edits at ${speaker.label ?? speaker.id}`
              : `Aim edits at ${speaker.label ?? speaker.id}`
          }
          title="Aim edits at this speaker"
          className={cn(
            'h-3 w-3 shrink-0 rounded-full transition-shadow',
            selected && 'ring-2 ring-brand ring-offset-1 ring-offset-canvas',
          )}
          style={{ background: speaker.color }}
        />

        <button
          type="button"
          disabled={!speaker.reference_url}
          onClick={() => togglePreview(speaker.reference_url)}
          aria-label={
            speaker.reference_url
              ? `Hear ${speaker.label ?? speaker.id} in the original audio`
              : 'No reference clip was saved for this speaker'
          }
          title={
            speaker.reference_url
              ? 'Hear this speaker in the original'
              : 'No reference clip was saved for this speaker'
          }
          className={cn(
            'grid h-6 w-6 shrink-0 place-items-center rounded-full transition-colors',
            speaker.reference_url
              ? 'bg-ink/[0.1] text-ink hover:bg-ink/[0.18]'
              : 'cursor-not-allowed bg-ink/[0.04] text-ink4',
          )}
        >
          {isPlaying ? <IconPause width={11} height={11} /> : <IconPlay width={11} height={11} />}
        </button>

        <div className="min-w-0">
          <SpeakerNameField speaker={speaker} />
          <div className="mt-0 flex flex-wrap items-center gap-x-1.5 gap-y-1 pl-1 text-[11.5px] text-ink4">
            <span>{fmtClock(speaker.speech_s)} of speech</span>
            {awaitingCast ? (
              <Select
                ariaLabel={`Cast mode for ${speaker.label ?? speaker.id}`}
                value={castMode}
                onValueChange={(m) =>
                  setCastMode(speaker.id, m as 'autodetect' | 'select')}
                className="h-6 w-[8.5rem] shrink-0 text-[11px]"
                items={[
                  { value: 'autodetect', label: 'Auto-detect' },
                  { value: 'select', label: 'Choose' },
                ]}
              />
            ) : null}
            {/* Casting is blocked until every row has a voice, so the rows
                holding it up have to be findable at a glance -- otherwise a
                disabled button on a twelve-speaker job is a guessing game. */}
            {awaitingCast && !(castMode === 'select' && voice) ? (
              <span className="shrink-0 rounded-md border border-warn/40 bg-warn/[0.08]
                               px-1.5 py-px text-[10.5px] font-medium text-warn">
                needs a voice
              </span>
            ) : null}
            {pickersEnabled ? (
              <Select
                ariaLabel={`Gender for ${speaker.label ?? speaker.id}`}
                value={genderRaw ?? undefined}
                onValueChange={onGender}
                placeholder="Gender"
                className="h-6 w-[9rem] shrink-0 text-[11px]"
                items={[
                  { value: 'male', label: 'Male' },
                  { value: 'female', label: 'Female' },
                ]}
              />
            ) : gender ? (
              <span className="rounded-md border border-line bg-ink/[0.05] px-1.5 py-px text-[11px] font-medium text-ink3">
                {gender}
              </span>
            ) : (
              <span className="text-[11px] text-ink4">gender unknown</span>
            )}
          </div>
          {cues && cues.length > 0 && (
            <div className="mt-0.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 pl-1">
              <span className="text-[10.5px] text-ink4">cues</span>
              {cues.slice(0, 10).map((c) => (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => requestSeek(c.start)}
                  title={`Go to cue #${c.cue_no} at ${fmtClock(c.start)}`}
                  className="rounded bg-brand/10 px-1.5 py-px text-[11px] font-medium text-brand
                             hover:bg-brand/20 transition-colors"
                >
                  {c.cue_no}
                </button>
              ))}
              {cues.length > 10 ? (
                <span className="text-[10.5px] text-ink4">+{cues.length - 10} more</span>
              ) : null}
            </div>
          )}
        </div>
      </div>

      <div className="ml-auto flex flex-wrap items-center justify-end gap-2">
        {rationale && !dirty ? (
          <Tooltip label={rationale}>
            <span className="cursor-help text-[11.5px] text-ink4">why this voice</span>
          </Tooltip>
        ) : null}

        <VoicePicker
          speaker={speaker}
          options={table.voice_options}
          value={voice}
          onChange={onVoice}
          disabled={!pickersEnabled}
        />

        {/* Absent, not disabled, when the engine would ignore it. */}
        {table.supports_tone ? (
          <Select
            ariaLabel={`Delivery for ${speaker.label ?? speaker.id}`}
            value={tone ?? undefined}
            onValueChange={onTone}
            placeholder="Delivery"
            className="h-8 w-[9.5rem]"
            disabled={!editable}
            // `value` is the wire value and is sent verbatim — "AIR style
            // news", not "AIR-style news". An unknown value is a 422.
            items={table.tone_options.map((t) => ({
              value: t.value,
              label: t.label,
              group: t.group,
            }))}
          />
        ) : null}

        <button
          type="button"
          onClick={onRevert}
          disabled={!dirty}
          aria-label={`Undo the unapplied change to ${speaker.label ?? speaker.id}`}
          className={cn(
            'grid h-9 w-9 place-items-center rounded-lg transition-colors',
            dirty ? 'text-warn hover:bg-warn/15' : 'cursor-not-allowed text-ink4/40',
          )}
        >
          <IconRevert width={14} height={14} />
        </button>
      </div>

      {/* What this speaker WILL read as, beneath the controls that set it:
          the old value struck through, the new one in blue. One row per
          field that is actually changing, so a voice swap does not print a
          gender line that says the same thing twice.

          This replaces a lone `was <voice>` note. Gender and tone had no
          equivalent at all, which mattered most for gender — it has no
          picker on this row, so the agent is the only way to change it and
          this is the only place the change can appear. */}
      {willChange ? (
        <dl className="w-full -mt-0.5 flex flex-wrap items-baseline gap-x-4 gap-y-1
                       pl-[3.5rem] text-[11.5px] sm:pl-0 sm:justify-end">
          {stagedGender && stagedGender !== currentGender ? (
            <div className="flex items-baseline gap-1.5">
              <dt className="text-ink4">gender</dt>
              <dd>
                <StagedValue
                  current={currentGender}
                  staged={stagedGender}
                  label={`Gender for ${speaker.label ?? speaker.id}`}
                />
              </dd>
            </div>
          ) : null}
          {stagedVoice && stagedVoice !== currentVoice ? (
            <div className="flex items-baseline gap-1.5">
              <dt className="text-ink4">voice</dt>
              <dd>
                <StagedValue
                  current={currentVoice}
                  staged={stagedVoice}
                  label={`Voice for ${speaker.label ?? speaker.id}`}
                />
              </dd>
            </div>
          ) : null}
          {stagedTone && stagedTone !== currentTone ? (
            <div className="flex items-baseline gap-1.5">
              <dt className="text-ink4">tone</dt>
              <dd>
                <StagedValue
                  current={currentTone}
                  staged={stagedTone}
                  label={`Delivery for ${speaker.label ?? speaker.id}`}
                />
              </dd>
            </div>
          ) : null}
        </dl>
      ) : null}
    </div>
  );
}
