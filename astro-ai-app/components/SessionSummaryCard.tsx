import React, { useMemo, useState } from 'react';
import { Pressable, Share, StyleSheet, Text, View } from 'react-native';
import { BlurView } from 'expo-blur';
import { useAppColors, shadowStyle } from '../lib/ui-theme';
import { useTranslation } from 'react-i18next';
import { useColorScheme } from '../hooks/use-color-scheme';

type SessionSummaryData = {
  focus: string;
  key_insight: string;
  mood_trend: string;
  reflection: string;
  next_step: string;
};

type Props = {
  summary: SessionSummaryData;
  userName?: string;
  onSave?: () => void;
  onClose?: () => void;
  saving?: boolean;
  variant?: 'default' | 'highlight';
  onFeedback?: (value: 'positive' | 'neutral' | 'negative') => void;
};

export default function SessionSummaryCard({
  summary,
  userName,
  onSave,
  onClose,
  saving,
  variant = 'default',
  onFeedback,
}: Props) {
  const c = useAppColors();
  const colorScheme = useColorScheme();
  const { t } = useTranslation();
  const isHighlight = variant === 'highlight';
  const [feedback, setFeedback] = useState<null | 'positive' | 'neutral' | 'negative'>(null);

  const feedbackTone = useMemo(() => {
    if (feedback === 'positive') {
      return { bg: c.isDark ? 'rgba(34,197,94,0.14)' : 'rgba(34,197,94,0.10)', border: c.isDark ? 'rgba(34,197,94,0.34)' : 'rgba(34,197,94,0.25)' };
    }
    if (feedback === 'negative') {
      return { bg: c.isDark ? 'rgba(239,68,68,0.14)' : 'rgba(239,68,68,0.10)', border: c.isDark ? 'rgba(239,68,68,0.34)' : 'rgba(239,68,68,0.25)' };
    }
    if (feedback === 'neutral') {
      return { bg: c.isDark ? 'rgba(148,163,184,0.12)' : 'rgba(148,163,184,0.10)', border: c.isDark ? 'rgba(148,163,184,0.28)' : 'rgba(148,163,184,0.20)' };
    }
    return null;
  }, [c.isDark, feedback]);

  const highlightShadow = {
    ...(shadowStyle(colorScheme) as any),
    ...(colorScheme === 'dark'
      ? { shadowOpacity: 0.34, shadowRadius: 20, elevation: 8 }
      : { shadowOpacity: 0.14, shadowRadius: 20, elevation: 6 }),
  };

  const onShare = async () => {
    const name = userName?.trim() || '';
    const message = `${t('summary.mySnapshot')}${name ? ` - ${name}` : ''}

🎯 ${summary.focus}
💬 ${summary.key_insight}
📈 ${summary.mood_trend}
🪞 ${summary.reflection}
🔮 ${summary.next_step}

${t('summary.shareFooter')}`;

    try {
      await Share.share({
        title: t('summary.shareTitle'),
        message,
      });
    } catch {
      // ignore
    }
  };

  return (
    <View style={styles.overlay}>
      <BlurView intensity={80} style={styles.backdrop}>
        <View
          style={[
            styles.card,
            {
              backgroundColor: isHighlight
                ? c.surfaceHighlight
                : c.card,
              borderWidth: isHighlight ? 1 : 0,
              borderColor: isHighlight
                ? c.accentBorderSoft
                : 'transparent',
            },
            isHighlight ? highlightShadow : shadowStyle(colorScheme),
          ]}
        >
          <Text style={[styles.title, { color: isHighlight ? c.primary : c.text }]}>{t('summary.title')}</Text>

          <View style={styles.section}>
            <Text style={[styles.label, { color: c.primary }]}>{t('summary.mainFocus')}</Text>
            <Text style={[styles.value, { color: c.text }]}>{summary.focus}</Text>
          </View>

          <View style={styles.section}>
            <Text style={[styles.label, { color: c.primary }]}>{t('summary.keyInsight')}</Text>
            <Text style={[styles.value, { color: c.text }]}>{summary.key_insight}</Text>
          </View>

          <View style={styles.section}>
            <Text style={[styles.label, { color: c.primary }]}>{t('summary.emotionalTrend')}</Text>
            <Text style={[styles.value, { color: c.text }]}>{summary.mood_trend}</Text>
          </View>

          <View style={styles.section}>
            <Text style={[styles.label, { color: c.primary }]}>{t('summary.reflection')}</Text>
            <Text style={[styles.value, { color: c.text }]}>{summary.reflection}</Text>
          </View>

          <View style={styles.section}>
            <Text style={[styles.label, { color: c.primary }]}>{t('summary.nextStep')}</Text>
            <Text style={[styles.value, { color: c.text }]}>{summary.next_step}</Text>
          </View>

          <View style={styles.actions}>
            {onSave && (
              <Pressable
                onPress={onSave}
                disabled={saving}
                style={[styles.button, styles.saveButton, { backgroundColor: c.primary }, saving && { opacity: 0.6 }]}
              >
                <Text style={styles.buttonText}>{saving ? t('summary.saving') : t('summary.save')}</Text>
              </Pressable>
            )}
            <Pressable
              onPress={onShare}
              style={[styles.button, styles.shareButton, { backgroundColor: c.card, borderColor: c.cardBorder }]}
            >
              <Text style={[styles.buttonText, { color: c.text }]}>{t('summary.share')}</Text>
            </Pressable>
            {onClose && (
              <Pressable onPress={onClose} style={[styles.button, styles.closeButton]}>
                <Text style={[styles.buttonText, { color: c.muted }]}>{t('summary.close')}</Text>
              </Pressable>
            )}
          </View>

          <View style={styles.feedbackWrap}>
            <Text style={[styles.feedbackTitle, { color: c.muted }]}>{t('summary.feedbackTitle')}</Text>
            <View style={styles.feedbackRow}>
              <Pressable
                onPress={() => {
                  setFeedback('positive');
                  onFeedback?.('positive');
                }}
                style={[styles.feedbackChip, { borderColor: c.border, backgroundColor: c.surfaceSubtle }]}
              >
                <Text style={[styles.feedbackText, { color: c.text }]}>{t('summary.feedbackHelpful')}</Text>
              </Pressable>
              <Pressable
                onPress={() => {
                  setFeedback('neutral');
                  onFeedback?.('neutral');
                }}
                style={[styles.feedbackChip, { borderColor: c.border, backgroundColor: c.surfaceSubtle }]}
              >
                <Text style={[styles.feedbackText, { color: c.text }]}>{t('summary.feedbackOkay')}</Text>
              </Pressable>
              <Pressable
                onPress={() => {
                  setFeedback('negative');
                  onFeedback?.('negative');
                }}
                style={[styles.feedbackChip, { borderColor: c.border, backgroundColor: c.surfaceSubtle }]}
              >
                <Text style={[styles.feedbackText, { color: c.text }]}>{t('summary.feedbackNotUseful')}</Text>
              </Pressable>
            </View>

            {feedbackTone ? (
              <View style={[styles.feedbackNote, { backgroundColor: feedbackTone.bg, borderColor: feedbackTone.border }]}>
                <Text style={[styles.feedbackNoteText, { color: c.muted }]}>{t('summary.feedbackSaved')}</Text>
              </View>
            ) : null}
          </View>
        </View>
      </BlurView>
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 9999,
  },
  backdrop: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  card: {
    width: '100%',
    maxWidth: 420,
    borderRadius: 20,
    padding: 20,
    gap: 16,
  },
  title: {
    fontSize: 22,
    fontWeight: '800',
    textAlign: 'center',
    marginBottom: 8,
  },
  section: {
    gap: 6,
  },
  label: {
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  value: {
    fontSize: 15,
    lineHeight: 21,
  },
  actions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 8,
    flexWrap: 'wrap',
  },
  button: {
    flex: 1,
    minWidth: 100,
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  saveButton: {},
  shareButton: {
    borderWidth: 1,
  },
  closeButton: {
    flex: 0,
    minWidth: 80,
  },
  buttonText: {
    color: 'white',
    fontWeight: '700',
    fontSize: 15,
  },

  feedbackWrap: {
    gap: 10,
    marginTop: 2,
  },
  feedbackTitle: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.4,
    textTransform: 'uppercase',
  },
  feedbackRow: {
    flexDirection: 'row',
    gap: 10,
    flexWrap: 'wrap',
  },
  feedbackChip: {
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 999,
    borderWidth: 1,
  },
  feedbackText: {
    fontWeight: '800',
    fontSize: 13,
  },
  feedbackNote: {
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderWidth: 1,
  },
  feedbackNoteText: {
    fontWeight: '700',
    fontSize: 13,
  },
});
