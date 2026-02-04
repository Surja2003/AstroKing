import { useColorScheme } from '../hooks/use-color-scheme';
import { cardShadowStyle, radius, spacing, padding } from './theme';

export function getAppColors(colorScheme) {
  const isDark = colorScheme === 'dark';

  const lightTheme = {
    // Base
    bg: '#F6F8FB',
    bg2: '#FFFFFF',
    card: '#FFFFFF',

    // Text
    text: '#0F172A',
    textSecondary: '#475569',
    muted: '#64748B',
    placeholder: '#94A3B8',

    // Inputs & Borders
    border: '#CBD5E1',
    cardBorder: '#CBD5E1',
    inputBg: '#FFFFFF',
    inputBorder: '#CBD5E1',

    // Accent system
    primary: '#5B6EFF',
    primarySoft: '#E0E7FF',
    onPrimary: '#FFFFFF',

    // Badge
    badgeBg: '#EEF2FF',
    badgeText: '#3730A3',
  };

  const darkTheme = {
    // Surfaces
    bg: '#0B0F1A',
    bg2: '#121826',
    card: 'rgba(255,255,255,0.06)',
    cardBorder: 'rgba(255,255,255,0.08)',

    // Text
    text: '#FFFFFF',
    textSecondary: '#CBD5E1',
    muted: '#A0A8C0',
    placeholder: '#94A3B8',

    // Borders & inputs
    border: 'rgba(255,255,255,0.08)',
    inputBg: 'rgba(255,255,255,0.06)',
    inputBorder: 'rgba(255,255,255,0.10)',

    // Brand
    primary: '#5B6EFF',
    primarySoft: 'rgba(91,110,255,0.20)',
    onPrimary: '#FFFFFF',

    // Badge
    badgeBg: 'rgba(91,110,255,0.18)',
    badgeText: '#A7B2FF',
  };

  return {
    isDark,

    ...(isDark ? darkTheme : lightTheme),

    // Derived interaction/surface tokens (kept here to avoid scattered hard-coded rgba).
    pressBg: isDark ? 'rgba(255,255,255,0.10)' : 'rgba(15,23,42,0.06)',
    surfaceHighlight: isDark ? 'rgba(255,255,255,0.10)' : lightTheme.card,
    surfaceSubtle: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(15,23,42,0.03)',
    surfaceSubtle2: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(15,23,42,0.02)',
    accentBorderSoft: isDark ? 'rgba(91,110,255,0.22)' : 'rgba(91,110,255,0.18)',
    accentBorderStrong: isDark ? 'rgba(91,110,255,0.40)' : 'rgba(91,110,255,0.22)',
    neutralMoodTint: isDark ? 'rgba(255,255,255,0.35)' : lightTheme.placeholder,
    placeholderSoft: isDark ? 'rgba(248,250,252,0.55)' : 'rgba(100,116,139,0.75)',

    // Kept for backwards compatibility / app usage
    primary2: '#22c55e',
    danger: '#ef4444',

    // Some components expect a badge border; derive it from the palette.
    badgeBorder: isDark ? 'rgba(91,110,255,0.35)' : lightTheme.inputBorder,
  };
}

export function useAppColors() {
  const colorScheme = useColorScheme();
  return getAppColors(colorScheme);
}

export function shadowStyle(colorScheme) {
  return cardShadowStyle(colorScheme);
}

export const tokens = {
  spacing,
  radius,
  padding,
};
