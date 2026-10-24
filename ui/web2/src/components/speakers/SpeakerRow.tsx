import { cn } from '../../lib/cn';
import { fmtClock, fmtRationale, genderWord } from '../../lib/format';
import { IconPause, IconPlay, IconRevert } from '../ui/Icons';
import { Tooltip } from '../ui/Tooltip';
import { Select } from '../ui/Select';
import { StagedValue } from '../ui/StagedValue';
import { VoicePicker } from './VoicePicker';
import { SpeakerNameField } from './SpeakerNameField';
import { togglePreview, useAudioPreview } from '../../hooks/useAudioPreview';
import type { AgentSpeaker, Speaker, SpeakerTable, VoiceOption } from '../../api/types';
import type { StagedChange } from '../../store/useStore';

/**
 * One speaker.
 *
 * Four things live on this row, and the difference between them matters:
 *
 *   name     EDITABLE, display only. Renaming dirties no stage — see
 *            `SpeakerNameField`.
 *   gender   READ-ONLY here — there is no picker for it. It is not a
 *            setting, it is the *reason* a particular voice was suggested,
 *            and showing it next to the picker is what makes the suggestion
 *            legible instead of arbitrary. It is also a prediction, and is
 *            labelled as one. It CAN be changed, but only by asking the
 *            agent ("make SPEAKER_00 female"), and a change asked for that
 *            way shows up here.
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
 * job's `pending.json`, which arrives as `fromAgent`. Gender only ever comes
 * from the second. A row that rendered only the first showed nothing at all
 * for a gender edit until it had been applied and the dub came back in a
 * different voice.
 */
export function SpeakerRow({
  speaker,
  table,
  staged,
  fromAgent,
  onVoice,
  onTone,
  onRevert,
  editable,
}: {
  speaker: Speaker;
  table: SpeakerTable;
  staged?: StagedChange;
  fromAgent?: AgentSpeaker | null;
  onVoice: (voice: string) => void;
  onTone: (tone: string) => void;
  onRevert: () => void;
  editable: boolean;
}) {
  const playing = useAudioPreview();
  const isPlaying = Boolean(speaker.reference_url && playing === speaker.reference_url);
  const dirty = Boolean(staged);

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

  const gender = genderWord(speaker.gender);
  const stagedGender = genderWord(fromAgent?.gender.staged ?? null);
  // Any of the three about to change, from either path.
  const willChange = Boolean(
    (stagedVoice && stagedVoice !== currentVoice)
    || (stagedTone && stagedTone !== currentTone)
    || (stagedGender && stagedGender !== gender),
  );

  const rationale = fmtRationale(speaker.voice_rationale);

  return (
    <div
      className={cn(
        'relative grid items-center gap-3 px-3.5 py-3',
        'grid-cols-1 sm:grid-cols-[minmax(0,1fr)_auto]',
        dirty && 'bg-warn/[0.06]',
      )}
    >
      {dirty ? (
        <span className="absolute inset-y-0 left-0 w-[3px] bg-warn" aria-hidden="true" />
      ) : null}

      <div className="flex min-w-0 items-center gap-2.5">
        <span
          className="h-2.5 w-2.5 shrink-0 rounded-full"
          style={{ background: speaker.color }}
          aria-hidden="true"
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
            'grid h-7 w-7 shrink-0 place-items-center rounded-full transition-colors',
            speaker.reference_url
              ? 'bg-ink/[0.1] text-ink hover:bg-ink/[0.18]'
              : 'cursor-not-allowed bg-ink/[0.04] text-ink4',
          )}
        >
          {isPlaying ? <IconPause width={11} height={11} /> : <IconPlay width={11} height={11} />}
        </button>

        <div className="min-w-0">
          <SpeakerNameField speaker={speaker} />
          <div className="mt-0.5 flex flex-wrap items-center gap-x-1.5 gap-y-1 pl-1 text-[11.5px] text-ink4">
            <span>{fmtClock(speaker.speech_s)} of speech</span>
            {speaker.n_segments ? (
              <>
                <span aria-hidden="true">·</span>
                <span>
                  {speaker.n_segments} line{speaker.n_segments === 1 ? '' : 's'}
                </span>
              </>
            ) : null}
            {gender ? (
              <Tooltip label="Detected from this speaker's reference clip. It is why the voices below are suggested — it is a prediction, and you can ignore it.">
                <span
                  className="cursor-help rounded-md border border-line bg-ink/[0.05] px-1.5
                             py-px text-[11px] font-medium text-ink3"
                >
                  {gender} · predicted
                </span>
              </Tooltip>
            ) : (
              <Tooltip label="The voice-casting stage records a predicted gender from the reference clip. It has not run for this speaker, so every voice is offered equally.">
                <span className="cursor-help text-ink4">gender not detected</span>
              </Tooltip>
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-end gap-2">
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
          disabled={!editable}
        />

        {/* Absent, not disabled, when the engine would ignore it. */}
        {table.supports_tone ? (
          <Select
            ariaLabel={`Delivery for ${speaker.label ?? speaker.id}`}
            value={tone ?? undefined}
            onValueChange={onTone}
            placeholder="Delivery"
            className="h-9 w-[9.5rem]"
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
        <dl className="col-span-full -mt-0.5 flex flex-wrap items-baseline gap-x-4 gap-y-1
                       pl-[3.5rem] text-[11.5px] sm:pl-0 sm:justify-end">
          {stagedGender && stagedGender !== gender ? (
            <div className="flex items-baseline gap-1.5">
              <dt className="text-ink4">gender</dt>
              <dd>
                <StagedValue
                  current={gender}
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
