import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Pressable, ScrollView, Share, StyleSheet, Text, View } from 'react-native';

import ScreenWrapper from '../components/ScreenWrapper';
import { api } from '../lib/api';
import { useUser } from '../context/user-context';
import { formatZodiacLabel, getZodiacFromDob } from '../lib/zodiac';
import { useTranslation } from 'react-i18next';
import { useAppColors, shadowStyle } from '../lib/ui-theme';
import { useColorScheme } from '../hooks/use-color-scheme';
import { padding, radius } from '../lib/theme';

type WeeklyTrendDay = {
  date: string;
  mood: string;
  counts?: Record<string, number>;
};

type WeeklyReportResponse = {
  days: number;
  range?: { from: string; to: string };
  total_messages?: number;
  active_days?: number;
  streak_days?: number;
  topics: Record<string, number>;
  moods: Record<string, number>;
  trend: WeeklyTrendDay[];
  most_asked: string;
  key_insight: string;
};

function moodEmoji(mood: string) {
  const m = String(mood || '').toLowerCase();
  if (m.includes('happy')) return '😊';
  if (m.includes('anxiety') || m.includes('stress')) return '😰';
  if (m.includes('sad')) return '😟';
  return '😐';
}

function topicKey(topic: string) {
  const k = String(topic || '').toLowerCase();
  if (k === 'money') return 'money';
  if (k === 'love') return 'love';
  if (k === 'career') return 'career';
  if (k === 'mind') return 'mind';
  if (k === 'health') return 'health';
  return 'general';
}

function normalizeCounts(map: Record<string, number>) {
  const entries = Object.entries(map || {}).filter(([, v]) => typeof v === 'number' && v > 0);
  entries.sort((a, b) => b[1] - a[1]);
  return entries;
}

function moodColor(mood: string, neutralTint: string) {
  const m = String(mood || '').toLowerCase();
  if (m.includes('happy')) return '#22c55e';
  if (m.includes('anxiety') || m.includes('stress')) return '#f59e0b';
  if (m.includes('sad')) return '#60a5fa';
  return neutralTint;
}

function dayLabel(iso: string) {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso.slice(5);
    return d.toLocaleDateString([], { weekday: 'short' });
  } catch {
    return iso.slice(5);
  }
}

export default function WeeklyReportScreen() {
  const { user, language } = useUser();
  const { t } = useTranslation();
  const c = useAppColors();
  const colorScheme = useColorScheme();
  const [days, setDays] = useState(7);
  const [data, setData] = useState<WeeklyReportResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const topicLabel = useCallback(
    (topic: string) => {
      const k = topicKey(topic);
      const emoji =
        k === 'money' ? '💰' : k === 'love' ? '❤️' : k === 'career' ? '🎯' : k === 'mind' ? '🧠' : k === 'health' ? '💪' : '✨';
      return `${emoji} ${t(`weekly.topics.${k}`)}`;
    },
    [t]
  );

  const zodiac = useMemo(() => {
    return getZodiacFromDob(user?.dob?.trim() || '');
  }, [user?.dob]);

  const zodiacLabel = useMemo(() => formatZodiacLabel(zodiac), [zodiac]);

  const load = async () => {
    const name = user?.name?.trim() || '';
    const dob = user?.dob?.trim() || '';

    if (!name || !dob) {
      setError(t('weekly.setProfileToUnlock'));
      setData(null);
      return;
    }

    setLoading(true);
    setError('');
    try {
      const res = await api.get<WeeklyReportResponse>(
        `/weekly-report?name=${encodeURIComponent(name)}&dob=${encodeURIComponent(dob)}&days=${days}&language=${encodeURIComponent(
          language || 'en'
        )}`
      );
      setData(res.data);
    } catch {
      setError(t('errors.weeklyLoad'));
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [days, user?.dob, user?.name]);

  const topics = useMemo(() => normalizeCounts(data?.topics ?? {}), [data?.topics]);
  const moods = useMemo(() => normalizeCounts(data?.moods ?? {}), [data?.moods]);
  const trend = useMemo(() => (data?.trend ?? []).slice(-Math.min(7, days)), [data?.trend, days]);

  const maxDayTotal = useMemo(() => {
    let max = 1;
    for (const d of trend) {
      const counts = d.counts ?? {};
      const total = Object.values(counts).reduce((a, b) => a + (typeof b === 'number' ? b : 0), 0);
      if (total > max) max = total;
    }
    return max;
  }, [trend]);

  const totalMessages = data?.total_messages ?? topics.reduce((sum, [, v]) => sum + v, 0);
  const activeDays = data?.active_days ?? 0;
  const streakDays = data?.streak_days ?? 0;

  const shareMessage = useMemo(() => {
    const rangeText = data?.range?.from && data?.range?.to ? `${data.range.from} → ${data.range.to}` : `Last ${days} days`;
    const header = `AstroKing • Weekly Report\n${rangeText}${zodiacLabel ? ` • ${zodiacLabel}` : ''}`;
    const metrics = `Streak: ${streakDays} days • Active: ${activeDays} days • Chats: ${totalMessages}`;
    const keyInsight = data?.key_insight ? `Key insight: ${data.key_insight}` : '';
    const mostAsked = data?.most_asked ? `Most asked: ${topicLabel(data.most_asked)}` : '';
    const topTopics = topics.length
      ? `Top topics: ${topics
          .slice(0, 3)
          .map(([k, v]) => `${topicLabel(k)} (${v})`)
          .join(', ')}`
      : '';
    return [header, '', metrics, mostAsked, topTopics, keyInsight ? `\n${keyInsight}` : ''].join('\n').trim();
  }, [activeDays, data?.key_insight, data?.most_asked, data?.range?.from, data?.range?.to, days, streakDays, totalMessages, topicLabel, topics, zodiacLabel]);

  const onShare = async () => {
    try {
      await Share.share({
        title: 'AstroKing — Weekly Report',
        message: shareMessage,
      });
    } catch {
      // ignore
    }
  };

  return (
    <ScreenWrapper>
      <ScrollView contentContainerStyle={{ paddingBottom: 24 }} showsVerticalScrollIndicator={false}>
        <Text style={[styles.title, { color: c.text }]}>{t('weekly.title')}</Text>
        <Text style={[styles.subTitle, { color: c.muted }]}>
          {zodiacLabel ? `${zodiacLabel}  •  ` : ''}
          {data?.range?.from && data?.range?.to
            ? `${data.range.from} → ${data.range.to}`
            : `Last ${days} ${t('weekly.lastDaysSuffix')}`}
        </Text>

        {error ? <Text style={[styles.error, { color: c.danger }]}>{error}</Text> : null}

        <View style={styles.row}>
          {[7, 14].map((d) => (
            <Pressable
              key={d}
              onPress={() => setDays(d)}
              style={[
                styles.chip,
                {
                  borderColor: c.border,
                  backgroundColor: c.surfaceSubtle,
                },
                d === days && {
                  backgroundColor: c.primarySoft,
                  borderColor: c.accentBorderStrong,
                },
              ]}
            >
              <Text style={[styles.chipText, { color: c.text }, d === days && { color: c.primary }]}>{d} days</Text>
            </Pressable>
          ))}
          <Pressable
            onPress={load}
            style={[
              styles.chip,
              {
                borderColor: c.border,
                backgroundColor: c.surfaceSubtle2,
              },
              loading && { opacity: 0.7 },
            ]}
            disabled={loading}
          >
            <Text style={[styles.chipText, { color: c.text }]}>{loading ? t('weekly.refreshing') : t('weekly.refresh')}</Text>
          </Pressable>
          <Pressable
            onPress={onShare}
            style={[
              styles.chip,
              {
                borderColor: c.border,
                backgroundColor: c.surfaceSubtle2,
              },
              (loading || !data) && { opacity: 0.6 },
            ]}
            disabled={loading || !data}
          >
            <Text style={[styles.chipText, { color: c.text }]}>{t('weekly.share')}</Text>
          </Pressable>
        </View>

        <View style={styles.metricsRow}>
          <View style={[styles.metricPill, { backgroundColor: c.card, borderColor: c.cardBorder }, shadowStyle(colorScheme)]}>
            <Text style={[styles.metricValue, { color: c.text }]}>{streakDays}</Text>
            <Text style={[styles.metricLabel, { color: c.muted }]}>{t('weekly.dayStreak')}</Text>
          </View>
          <View style={[styles.metricPill, { backgroundColor: c.card, borderColor: c.cardBorder }, shadowStyle(colorScheme)]}>
            <Text style={[styles.metricValue, { color: c.text }]}>{activeDays}</Text>
            <Text style={[styles.metricLabel, { color: c.muted }]}>{t('weekly.activeDays')}</Text>
          </View>
          <View style={[styles.metricPill, { backgroundColor: c.card, borderColor: c.cardBorder }, shadowStyle(colorScheme)]}>
            <Text style={[styles.metricValue, { color: c.text }]}>{totalMessages}</Text>
            <Text style={[styles.metricLabel, { color: c.muted }]}>{t('weekly.chats')}</Text>
          </View>
        </View>

        <View
          style={[
            styles.card,
            {
              backgroundColor: c.card,
              borderColor: c.cardBorder,
              borderLeftColor: c.primary,
            },
            shadowStyle(colorScheme),
          ]}
        >
          <View style={styles.cardHeaderRow}>
            <Text style={[styles.cardTitle, { color: c.text }]}>{t('weekly.dailyEnergyPattern')}</Text>
            <Text style={[styles.muted, { color: c.muted }]}>{t('weekly.lastDays', { count: Math.min(7, days) })}</Text>
          </View>
          {trend.length ? (
            <View style={styles.chartRow}>
              {trend.map((d) => {
                const counts = d.counts ?? {};
                const total = Object.values(counts).reduce(
                  (a, b) => a + (typeof b === 'number' ? b : 0),
                  0
                );
                const height = 10 + Math.round((total / maxDayTotal) * 34);
                const color = moodColor(d.mood, c.neutralMoodTint);
                return (
                  <View key={d.date} style={styles.chartCol}>
                    <View style={[styles.chartBar, { height, backgroundColor: color }]} />
                    <Text style={[styles.chartLabel, { color: c.muted }]}>{dayLabel(d.date)}</Text>
                    <Text style={[styles.chartEmoji, { color: c.text }]}>{moodEmoji(d.mood)}</Text>
                  </View>
                );
              })}
            </View>
          ) : (
            <Text style={[styles.muted, { color: c.muted }]}>{t('weekly.buildPattern')}</Text>
          )}
          <View style={styles.legendRow}>
            <Text style={[styles.legendItem, { color: c.muted }]}>😊 {t('weekly.positive')}</Text>
            <Text style={[styles.legendItem, { color: c.muted }]}>😰 {t('weekly.anxious')}</Text>
            <Text style={[styles.legendItem, { color: c.muted }]}>😟 {t('weekly.low')}</Text>
            <Text style={[styles.legendItem, { color: c.muted }]}>😐 {t('weekly.neutral')}</Text>
          </View>
        </View>

        <View
          style={[
            styles.card,
            {
              backgroundColor: c.card,
              borderColor: c.cardBorder,
              borderLeftColor: c.primary,
            },
            shadowStyle(colorScheme),
          ]}
        >
          <Text style={[styles.cardTitle, { color: c.text }]}>{t('weekly.keyInsight')}</Text>
          <Text style={[styles.cardBody, { color: c.text }]}>
            {loading ? t('common.loading') : data?.key_insight || t('weekly.noInsightYet')}
          </Text>
        </View>

        <View
          style={[
            styles.card,
            {
              backgroundColor: c.card,
              borderColor: c.cardBorder,
              borderLeftColor: c.primary,
            },
            shadowStyle(colorScheme),
          ]}
        >
          <Text style={[styles.cardTitle, { color: c.text }]}>{t('weekly.mostAsked')}</Text>
          <Text style={[styles.cardBody, { color: c.text }]}>{data?.most_asked ? topicLabel(data.most_asked) : '—'}</Text>
        </View>

        <View style={styles.grid}>
          <View
            style={[
              styles.halfCard,
              {
                backgroundColor: c.card,
                borderColor: c.cardBorder,
                borderLeftColor: c.primary,
              },
              shadowStyle(colorScheme),
            ]}
          >
            <Text style={[styles.cardTitle, { color: c.text }]}>{t('weekly.topicMix')}</Text>
            {topics.length ? (
              topics.slice(0, 5).map(([k, v]) => (
                <Text key={k} style={[styles.listItem, { color: c.text }]}>
                  {topicLabel(k)}: {v}{totalMessages ? ` (${Math.round((v / Math.max(1, totalMessages)) * 100)}%)` : ''}
                </Text>
              ))
            ) : (
              <Text style={[styles.muted, { color: c.muted }]}>{t('weekly.noMessages')}</Text>
            )}
          </View>

          <View
            style={[
              styles.halfCard,
              {
                backgroundColor: c.card,
                borderColor: c.cardBorder,
                borderLeftColor: c.primary,
              },
              shadowStyle(colorScheme),
            ]}
          >
            <Text style={[styles.cardTitle, { color: c.text }]}>{t('weekly.moodTrend')}</Text>
            {moods.length ? (
              moods.slice(0, 5).map(([k, v]) => (
                <Text key={k} style={[styles.listItem, { color: c.text }]}>
                  {moodEmoji(k)} {k}: {v}
                </Text>
              ))
            ) : (
              <Text style={[styles.muted, { color: c.muted }]}>{t('weekly.noMoodData')}</Text>
            )}
          </View>
        </View>

        <View
          style={[
            styles.card,
            {
              backgroundColor: c.card,
              borderColor: c.cardBorder,
              borderLeftColor: c.primary,
            },
            shadowStyle(colorScheme),
          ]}
        >
          <Text style={[styles.cardTitle, { color: c.text }]}>{t('weekly.dailySnapshot')}</Text>
          {data?.trend?.length ? (
            data.trend.slice(-7).map((d) => (
              <View key={d.date} style={[styles.dayRow, { borderTopColor: c.border }]}>
                <Text style={[styles.dayDate, { color: c.muted }]}>{d.date}</Text>
                <Text style={[styles.dayMood, { color: c.text }]}>
                  {moodEmoji(d.mood)} {d.mood}
                </Text>
              </View>
            ))
          ) : (
            <Text style={[styles.muted, { color: c.muted }]}>{t('weekly.populate')}</Text>
          )}
        </View>
      </ScrollView>
    </ScreenWrapper>
  );
}

const styles = StyleSheet.create({
  title: { fontSize: 26, fontWeight: '900', marginBottom: 8 },
  subTitle: { marginBottom: 14 },
  error: { marginBottom: 10 },

  row: { flexDirection: 'row', gap: 10, flexWrap: 'wrap', marginBottom: 12 },
  chip: {
    borderWidth: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 999,
  },
  chipText: { fontWeight: '900' },

  card: {
    borderWidth: 1,
    borderLeftWidth: 2,
    borderRadius: radius.card,
    padding: padding.card,
    marginBottom: 12,
  },
  cardTitle: { fontWeight: '900', marginBottom: 8, fontSize: 18 },
  cardBody: { lineHeight: 20, fontSize: 14 },
  muted: {},

  metricsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
    flexWrap: 'wrap',
  },
  metricPill: {
    flexGrow: 1,
    flexBasis: 110,
    borderWidth: 1,
    borderRadius: radius.card,
    paddingVertical: 12,
    paddingHorizontal: 12,
    alignItems: 'center',
  },
  metricValue: { fontWeight: '900', fontSize: 18 },
  metricLabel: { marginTop: 4, fontWeight: '800' },

  cardHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginBottom: 10,
  },
  chartRow: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  chartCol: { alignItems: 'center', minWidth: 38 },
  chartBar: {
    width: 22,
    borderRadius: 10,
    marginBottom: 8,
  },
  chartLabel: { fontWeight: '800', fontSize: 12 },
  chartEmoji: { marginTop: 2 },
  legendRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  legendItem: { fontWeight: '800', fontSize: 12 },

  grid: { flexDirection: 'row', gap: 12, flexWrap: 'wrap' },
  halfCard: {
    flexGrow: 1,
    flexBasis: 160,
    borderWidth: 1,
    borderLeftWidth: 2,
    borderRadius: radius.card,
    padding: padding.card,
    marginBottom: 12,
  },
  listItem: { marginBottom: 6, fontWeight: '700' },

  dayRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderTopWidth: 1,
  },
  dayDate: { fontWeight: '800' },
  dayMood: { fontWeight: '800' },
});
