import { cn } from '../../lib/cn';
import { fmtClock, fmtRationale, genderWord } from '../../lib/format';
import { IconChevron, IconPause, IconPlay, IconRevert } from '../frontend/Icons';
import { Tooltip } from '../frontend/Tooltip';
import { Select } from '../frontend/Select';
import { Eyebrow } from '../frontend/Card';
import { StagedValue } from '../frontend/StagedValue';
import { VoicePicker } from './VoicePicker';
import { togglePreview, useAudioPreview } from '../../hooks/useAudioPreview';
import { useStore } from '../../store/useStore';
import type { AgentScriptRow, AgentSpeaker, Speaker, SpeakerTable, VoiceOption } from '../../api/types';
import type { StagedChange } from '../../store/useStore';

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
  expanded,
  onToggle,
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
  expanded: boolean;
  onToggle: () => void;
}) {
  const requestSeek = useStore((s) => s.requestSeek);
  const selectedSpeaker = useStore((s) => s.selectedSpeaker);
  const setSelectedSpeaker = useStore((s) => s.setSelectedSpeaker);
  const setSpeakerPhoto = useStore((s) => s.setSpeakerPhoto);
  const selected = selectedSpeaker === speaker.id;
  const playing = useAudioPreview();
  const isPlaying = Boolean(speaker.reference_url && playing === speaker.reference_url);
  const dirty = Boolean(staged);

  const awaitingCast = useStore((s) => Boolean(s.progress?.milestones?.awaiting_cast));
  const pickersEnabled = editable;

  const voice = staged?.voice !== undefined ? staged.voice : speaker.voice;
  const tone = staged?.tone !== undefined ? staged.tone : speaker.tone;

  const voiceLabel = (v: string | null | undefined) =>
    (v
      ? (table.voice_options.find((o: VoiceOption) => o.voice === v)?.display
        ?? v.split('-').pop() ?? v)
      : null);
  const toneLabel = (t: string | null | undefined) =>
    (t ? table.tone_options.find((o) => o.value === t)?.label ?? t : null);

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

  const willChange = Boolean(
    (stagedVoice && stagedVoice !== currentVoice)
    || (stagedTone && stagedTone !== currentTone)
    || (stagedGender && stagedGender !== currentGender),
  );

  const rationale = fmtRationale(speaker.voice_rationale);
  const cueCount = cues?.length ?? 0;
  const displayVoice = voiceLabel(voice);
  const hasVoice = Boolean(voice);

  return (
    <div
      className={cn(
        'relative',
        dirty && 'bg-warn/[0.06]',
        selected && 'ring-2 ring-inset ring-brand',
      )}
    >
      {dirty ? (
        <span className="absolute inset-y-0 left-0 w-[3px] bg-warn" aria-hidden="true" />
      ) : null}

      {/* ── Collapsed header row ── */}
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors
                   hover:bg-ink/[0.03]"
      >
        <span
          className="h-2.5 w-2.5 shrink-0 rounded-full"
          style={{ background: speaker.color }}
          onClick={(e) => {
            e.stopPropagation();
            setSelectedSpeaker(selected ? null : speaker.id);
          }}
        />

        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <span className="text-[15px] font-semibold text-ink">
              {speaker.label ?? speaker.id}
            </span>
            <span className="text-[13px] text-ink3">
              {fmtClock(speaker.speech_s)} of speech &middot; {cueCount} cue{cueCount !== 1 ? 's' : ''}
            </span>
          </div>

          {!expanded && hasVoice ? (
            <div className="mt-0.5 flex items-center gap-1.5 text-[13px] text-ink2">
              <button
                type="button"
                disabled={!speaker.reference_url}
                onClick={(e) => {
                  e.stopPropagation();
                  togglePreview(speaker.reference_url);
                }}
                className={cn(
                  'inline-flex items-center gap-1 transition-colors',
                  speaker.reference_url
                    ? 'text-ink2 hover:text-ink'
                    : 'cursor-not-allowed text-ink4',
                )}
              >
                {isPlaying
                  ? <IconPause width={11} height={11} />
                  : <IconPlay width={11} height={11} />}
              </button>
              <span>{gender ? `${gender} voice` : 'Voice'} &middot; Preview</span>
            </div>
          ) : !expanded ? (
            <p className="mt-0.5 text-[13px] text-ink3">
              Expand to change gender, voice, delivery style and inspect assigned cues.
            </p>
          ) : null}
        </div>

        <div className="flex items-center gap-2">
          {hasVoice ? (
            <span className="rounded-md border border-ok/40 bg-ok/10 px-2 py-0.5
                             text-[12px] font-medium text-ok">
              Assigned
            </span>
          ) : awaitingCast ? (
            <span className="rounded-md border border-warn/40 bg-warn/[0.08] px-2 py-0.5
                             text-[12px] font-medium text-warn">
              Needs voice
            </span>
          ) : null}

          <IconChevron
            width={14} height={14}
            className={cn(
              'shrink-0 text-ink3 transition-transform',
              expanded && 'rotate-180',
            )}
          />
        </div>
      </button>

      {/* ── Expanded detail card ── */}
      {expanded ? (
        <div className="border-t border-line bg-canvas/40 px-4 pb-3 pt-2.5">
          {/* Voice preview row */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={!speaker.reference_url}
              onClick={() => togglePreview(speaker.reference_url)}
              aria-label={
                speaker.reference_url
                  ? `Hear ${speaker.label ?? speaker.id} in the original audio`
                  : 'No reference clip was saved for this speaker'
              }
              className={cn(
                'grid h-7 w-7 shrink-0 place-items-center rounded-full transition-colors',
                speaker.reference_url
                  ? 'bg-ink/[0.1] text-ink hover:bg-ink/[0.18]'
                  : 'cursor-not-allowed bg-ink/[0.04] text-ink4',
              )}
            >
              {isPlaying ? <IconPause width={12} height={12} /> : <IconPlay width={12} height={12} />}
            </button>
            <div className="min-w-0 text-[14px]">
              <span className="font-medium text-ink">{displayVoice ?? 'Not assigned'}</span>
              <span className="ml-1 text-ink3">Selected voice</span>
            </div>
          </div>

          {/* Controls */}
          <div className="mt-3 space-y-2.5">
            {speaker.face_crop_urls.length > 0 ? (
              <div>
                <Eyebrow className="mb-1.5">Photo</Eyebrow>
                <div className="flex gap-2">
                  {speaker.face_crop_urls.map((url, i) => (
                    <button
                      key={url}
                      type="button"
                      disabled={!editable}
                      onClick={() => setSpeakerPhoto(speaker.id, i)}
                      aria-label={`Use candidate photo ${i + 1} for ${speaker.label ?? speaker.id}`}
                      aria-pressed={(speaker.face_photo_index ?? 0) === i}
                      className={cn(
                        'h-14 w-14 shrink-0 overflow-hidden rounded-lg border-2 transition-colors',
                        (speaker.face_photo_index ?? 0) === i
                          ? 'border-brand'
                          : 'border-line hover:border-ink3',
                        !editable && 'cursor-not-allowed opacity-60',
                      )}
                    >
                      <img src={url} alt="" className="h-full w-full object-cover" />
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            <div>
              <Eyebrow className="mb-1.5">Voice gender</Eyebrow>
              {pickersEnabled ? (
                <Select
                  ariaLabel={`Gender for ${speaker.label ?? speaker.id}`}
                  value={genderRaw ?? undefined}
                  onValueChange={onGender}
                  placeholder="Gender"
                  items={[
                    { value: 'male', label: 'Male' },
                    { value: 'female', label: 'Female' },
                  ]}
                />
              ) : gender ? (
                <span className="rounded-md border border-line bg-ink/[0.05] px-2 py-1 text-[12.5px] font-medium text-ink2">
                  {gender}
                </span>
              ) : (
                <span className="text-[12px] text-ink4">Gender unknown</span>
              )}
            </div>

            <div>
              <Eyebrow className="mb-1.5">Voice</Eyebrow>
              <VoicePicker
                speaker={speaker}
                options={table.voice_options}
                value={voice}
                onChange={onVoice}
                disabled={!pickersEnabled}
                target={table.language}
              />
            </div>

            {table.supports_tone ? (
              <div>
                <Eyebrow className="mb-1.5">Delivery style</Eyebrow>
                <Select
                  ariaLabel={`Delivery for ${speaker.label ?? speaker.id}`}
                  value={tone ?? undefined}
                  onValueChange={onTone}
                  placeholder="Delivery"
                  disabled={!editable}
                  items={table.tone_options.map((t) => ({
                    value: t.value,
                    label: t.label,
                    group: t.group,
                  }))}
                />
              </div>
            ) : null}
          </div>

          {/* Staged change summary */}
          {willChange ? (
            <dl className="mt-3 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[11.5px]">
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

          {rationale && !dirty ? (
            <Tooltip label={rationale}>
              <span className="mt-2 inline-block cursor-help text-[12.5px] text-ink4">
                Why this voice
              </span>
            </Tooltip>
          ) : null}

          {dirty ? (
            <button
              type="button"
              onClick={onRevert}
              className="mt-2 inline-flex items-center gap-1.5 text-[13px] font-medium
                         text-warn transition-colors hover:text-warn/80"
            >
              <IconRevert width={13} height={13} />
              Undo changes
            </button>
          ) : null}

          {/* Assigned cues */}
          {cues && cueCount > 0 ? (
            <div className="mt-4">
              <div className="flex items-baseline justify-between">
                <span className="text-[14px] font-semibold text-ink">Assigned cues</span>
                <span className="text-[13px] text-ink3">View all {cueCount}</span>
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {cues.slice(0, 12).map((c) => (
                  <button
                    key={c.id}
                    type="button"
                    onClick={() => requestSeek(c.start)}
                    title={`Go to cue #${c.cue_no} at ${fmtClock(c.start)}`}
                    className="rounded-md bg-brand/10 px-2.5 py-1 text-[13px] font-medium
                               text-brand transition-colors hover:bg-brand/20"
                  >
                    #{c.cue_no}
                  </button>
                ))}
                {cueCount > 12 ? (
                  <span className="rounded-md bg-ink/[0.05] px-2.5 py-1 text-[13px] text-ink4">
                    +{cueCount - 12} more
                  </span>
                ) : null}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
