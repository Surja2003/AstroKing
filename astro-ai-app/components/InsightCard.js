import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useColorScheme } from '../hooks/use-color-scheme';
import { getAppColors, shadowStyle } from '../lib/ui-theme';
import { padding, radius } from '../lib/theme';

export default function InsightCard({ title, body, footer = '' }) {
  const colorScheme = useColorScheme();
  const c = getAppColors(colorScheme);

  return (
    <View
      style={[
        styles.card,
        {
          backgroundColor: c.card,
          borderColor: c.cardBorder,
        },
        shadowStyle(colorScheme),
      ]}
    >
      {!!title && <Text style={[styles.title, { color: c.text }]}>{title}</Text>}
      {!!body && <Text style={[styles.body, { color: c.muted }]}>{body}</Text>}
      {!!footer && <Text style={[styles.footer, { color: c.muted }]}>{footer}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderWidth: 1,
    borderRadius: radius.card,
    padding: padding.card,
    gap: 8,
  },
  title: {
    fontSize: 16,
    fontWeight: '800',
    letterSpacing: 0.2,
  },
  body: {
    fontSize: 14,
    lineHeight: 20,
  },
  footer: {
    marginTop: 6,
    fontSize: 12,
    lineHeight: 16,
  },
});
