import { Platform } from 'react-native';

export const spacing = {
  xs: 6,
  sm: 10,
  md: 16,
  lg: 22,
  xl: 28,
};

export const radius = {
  card: 18,
  input: 24,
  pill: 30,
  bubble: 18,
};

export const padding = {
  card: 16,
  bubble: 14,
};

export const shadow = {
  card: {
    dark: {
      shadowColor: '#000',
      shadowOpacity: 0.28,
      shadowRadius: 16,
      shadowOffset: { width: 0, height: 10 },
      elevation: 6,
      webBoxShadow: '0px 18px 44px rgba(0,0,0,0.55)',
    },
    light: {
      shadowColor: '#000',
      shadowOpacity: 0.10,
      shadowRadius: 16,
      shadowOffset: { width: 0, height: 10 },
      elevation: 4,
      webBoxShadow: '0px 16px 40px rgba(2,6,23,0.12)',
    },
  },
};

export function cardShadowStyle(colorScheme) {
  const isDark = colorScheme === 'dark';

  if (Platform.OS === 'web') {
    return {
      boxShadow: isDark ? shadow.card.dark.webBoxShadow : shadow.card.light.webBoxShadow,
    };
  }

  const s = isDark ? shadow.card.dark : shadow.card.light;
  return {
    shadowColor: s.shadowColor,
    shadowOpacity: s.shadowOpacity,
    shadowRadius: s.shadowRadius,
    shadowOffset: s.shadowOffset,
    elevation: s.elevation,
  };
}

export function accentGlowStyle(colorScheme, color) {
  const isDark = colorScheme === 'dark';
  const glow = isDark ? 0.42 : 0.22;
  const accent = color || '#5B6CFF';

  if (Platform.OS === 'web') {
    return {
      boxShadow: `0px 14px 38px rgba(2,6,23,${isDark ? 0.55 : 0.14}), 0px 0px 22px rgba(91,108,255,${glow})`,
    };
  }

  return {
    shadowColor: accent,
    shadowOpacity: glow,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 10 },
    elevation: 7,
  };
}
