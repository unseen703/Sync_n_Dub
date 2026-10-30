/** Indian-language proximity groups. Codes are what VoiceOption.language carries
 * (ISO 639-1 where it exists, else 639-3). Mirrors INDIAN_LANGUAGE_GROUPS. */
export const LANG_GROUPS: string[][] = [
  ['ta', 'ml'],                                            // 0 South A
  ['te', 'kn'],                                            // 1 South B
  ['hi', 'gu', 'mr', 'kok', 'ur'],                         // 2 Central/Western
  ['pa', 'sd', 'ks'],                                      // 3 Northern
  ['or', 'mai', 'bn', 'as', 'brx',
   'mni', 'ne', 'sa', 'sat', 'doi'],                       // 4 Eastern/NE + other scheduled
];

// South Indian "region" = groups 0 + 1. Every other group is its own region.
const REGIONS: string[][][] = [
  [LANG_GROUPS[0], LANG_GROUPS[1]],
  [LANG_GROUPS[2]],
  [LANG_GROUPS[3]],
  [LANG_GROUPS[4]],
];

function base(code: string | null | undefined): string {
  return (code ?? '').split('-')[0].toLowerCase();
}

/** 0 = exact target, 1 = same subgroup, 2 = same region, 3 = everything else. */
export function langPriority(voiceLang: string | null | undefined, target: string | null | undefined): number {
  const v = base(voiceLang);
  const t = base(target);
  if (!t || !v) return 3;
  if (v === t) return 0;
  const subgroup = LANG_GROUPS.find((g) => g.includes(t));
  if (subgroup?.includes(v)) return 1;
  const region = REGIONS.find((r) => r.some((g) => g.includes(t)));
  if (region?.some((g) => g.includes(v))) return 2;
  return 3;
}
