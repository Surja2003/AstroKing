import React, { useEffect, useMemo, useState } from 'react';
import { FlatList, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';

import ScreenWrapper from '../components/ScreenWrapper';
import { api } from '../lib/api';
import { useUser } from '../context/user-context';
import { useBackendHealth } from '../hooks/use-backend-health';
import { useTranslation } from 'react-i18next';
import { useAppColors, shadowStyle } from '../lib/ui-theme';
import { useColorScheme } from '../hooks/use-color-scheme';
import { padding, radius } from '../lib/theme';
import { loadLocalPalmHistory } from '../lib/palm-history';

type ChatItem = {
  id: number;
  message: string;
  reply: string;
  emotion?: string | null;
  created_at?: string | null;
};

type PalmItem = {
  id: number | string;
  image_path?: string;
  image_url?: string | null;
  created_at?: string | null;
  local_uri?: string | null;
  message?: string | null;
};

type SummaryItem = {
  id: number;
  focus: string;
  key_insight: string;
  mood_trend: string;
  reflection: string;
  next_step: string;
  created_at?: string | null;
};

type ChatHistoryResponse = {
  items: ChatItem[];
};

type PalmHistoryResponse = {
  items: PalmItem[];
};

type SummaryHistoryResponse = {
  items: SummaryItem[];
};

export default function HistoryScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ name?: string; dob?: string }>();
  const { user } = useUser();
  const { t } = useTranslation();
  const c = useAppColors();
  const colorScheme = useColorScheme();

  const name = useMemo(() => {
    return (params?.name ?? user?.name ?? '').toString().trim();
  }, [params?.name, user?.name]);

  const dob = useMemo(() => {
    return (params?.dob ?? user?.dob ?? '').toString().trim();
  }, [params?.dob, user?.dob]);

  const [chats, setChats] = useState<ChatItem[]>([]);
  const [palms, setPalms] = useState<PalmItem[]>([]);
  const [summaries, setSummaries] = useState<SummaryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { serverOk } = useBackendHealth();

  const apiBaseUrl = useMemo(() => {
    const base = (api as any)?.defaults?.baseURL ?? '';
    return typeof base === 'string' ? base.replace(/\/$/, '') : '';
  }, []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      setError('');
      setLoading(true);
      try {
        if (!name || !dob) {
          setChats([]);
          setPalms([]);
          setSummaries([]);
          return;
        }

        const [chatRes, palmRes, summaryRes, localPalms] = await Promise.all([
          api.get<ChatHistoryResponse>(`/history/chat?name=${encodeURIComponent(name)}&dob=${encodeURIComponent(dob)}&limit=10`),
          api.get<PalmHistoryResponse>(`/history/palms?name=${encodeURIComponent(name)}&dob=${encodeURIComponent(dob)}&limit=5`),
          api.get<SummaryHistoryResponse>(`/history/summaries?name=${encodeURIComponent(name)}&dob=${encodeURIComponent(dob)}&limit=10`),
          loadLocalPalmHistory(name, dob, 10),
        ]);

        if (cancelled) return;
        setChats(chatRes.data?.items ?? []);
        const remote = palmRes.data?.items ?? [];
        const local = (localPalms ?? []).map((p) => ({
          id: p.id,
          created_at: p.created_at,
          local_uri: p.local_uri,
          message: p.message,
        } as PalmItem));

        // Show most recent first, mixing local + remote.
        const merged = [...local, ...remote].sort((a, b) => String(b.created_at ?? '').localeCompare(String(a.created_at ?? '')));
        setPalms(merged);
        setSummaries(summaryRes.data?.items ?? []);
      } catch {
        if (!cancelled) setError(t('errors.historyLoad'));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [dob, name, t]);

  const formatStamp = (iso?: string | null) => {
    if (!iso) return '';
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return '';
      return d.toLocaleString([], { month: 'short', day: '2-digit', hour: 'numeric', minute: '2-digit' });
    } catch {
      return '';
    }
  };

  const emotionBadge = (emotion?: string | null) => {
    const e = String(emotion ?? '').toLowerCase();
    if (!e) return null;
    if (e.includes('happy')) return { emoji: '😊', label: t('history.moodPositive'), tint: '#22c55e' };
    if (e.includes('anxiety') || e.includes('stress')) return { emoji: '😰', label: t('history.moodAnxious'), tint: '#f59e0b' };
    if (e.includes('sad')) return { emoji: '😟', label: t('history.moodLow'), tint: '#60a5fa' };
    return { emoji: '😐', label: t('history.moodNeutral'), tint: c.placeholder };
  };

  const statusTone = serverOk
    ? {
        bg: c.isDark ? 'rgba(34,197,94,0.14)' : 'rgba(34,197,94,0.10)',
        border: c.isDark ? 'rgba(34,197,94,0.34)' : 'rgba(34,197,94,0.25)',
        text: '#22c55e',
        label: t('common.connected'),
      }
    : {
        bg: c.isDark ? 'rgba(239,68,68,0.14)' : 'rgba(239,68,68,0.10)',
        border: c.isDark ? 'rgba(239,68,68,0.34)' : 'rgba(239,68,68,0.25)',
        text: c.danger,
        label: t('common.serverOffline'),
      };


  return (
    <ScreenWrapper>
      <ScrollView contentContainerStyle={{ paddingBottom: 24 }} showsVerticalScrollIndicator={false}>
        <Text style={[styles.title, { color: c.text }]}>{t('history.title')}</Text>
        <Text style={[styles.subTitle, { color: c.muted }]}>{t('history.subtitle')}</Text>

        <View
          style={[
            styles.statusPill,
            {
              backgroundColor: statusTone.bg,
              borderColor: statusTone.border,
            },
          ]}
        >
          <Text style={[styles.statusText, { color: statusTone.text }]}>{statusTone.label}</Text>
        </View>

        {!name || !dob ? <Text style={[styles.muted, { color: c.muted }]}>{t('history.loginFirst')}</Text> : null}

        {loading ? <Text style={[styles.muted, { color: c.muted }]}>{t('common.loading')}</Text> : null}
        {error ? <Text style={[styles.error, { color: c.danger }]}>{error}</Text> : null}

        <Text style={[styles.section, { color: c.primary }]}>{t('history.snapshots')}</Text>
        <FlatList
          data={summaries}
          keyExtractor={(item) => String(item.id)}
          scrollEnabled={false}
          ItemSeparatorComponent={() => <View style={{ height: 12 }} />}
          ListEmptyComponent={!loading && name && dob ? <Text style={[styles.muted, { color: c.muted }]}>{t('history.noSnapshots')}</Text> : null}
          renderItem={({ item }) => {
            const stamp = formatStamp(item.created_at);
            return (
              <View style={[styles.snapshotCard, { backgroundColor: c.card, borderColor: c.cardBorder }, shadowStyle(colorScheme)]}>
                <View style={styles.cardTop}>
                  <Text style={[styles.snapshotTitle, { color: c.primary }]}>{t('history.snapshotTitle')}</Text>
                  {!!stamp && <Text style={[styles.stamp, { color: c.muted }]}>{stamp}</Text>}
                </View>
                <View style={styles.snapshotRow}>
                  <Text style={[styles.snapshotLabel, { color: c.muted }]}>{t('summary.mainFocus')}:</Text>
                  <Text style={[styles.snapshotValue, { color: c.text }]}>{item.focus}</Text>
                </View>
                <View style={styles.snapshotRow}>
                  <Text style={[styles.snapshotLabel, { color: c.muted }]}>{t('summary.keyInsight')}:</Text>
                  <Text style={[styles.snapshotValue, { color: c.text }]}>{item.key_insight}</Text>
                </View>
                <View style={styles.snapshotRow}>
                  <Text style={[styles.snapshotLabel, { color: c.muted }]}>{t('summary.emotionalTrend')}:</Text>
                  <Text style={[styles.snapshotValue, { color: c.text }]}>{item.mood_trend}</Text>
                </View>
                <View style={styles.snapshotRow}>
                  <Text style={[styles.snapshotLabel, { color: c.muted }]}>{t('summary.reflection')}:</Text>
                  <Text style={[styles.snapshotValue, { color: c.text }]}>{item.reflection}</Text>
                </View>
                <View style={styles.snapshotRow}>
                  <Text style={[styles.snapshotLabel, { color: c.muted }]}>{t('summary.nextStep')}:</Text>
                  <Text style={[styles.snapshotValue, { color: c.text }]}>{item.next_step}</Text>
                </View>
              </View>
            );
          }}
        />

        <Text style={[styles.section, { color: c.primary }]}>{t('history.chatHistory')}</Text>
        <FlatList
          data={chats}
          keyExtractor={(item) => String(item.id)}
          scrollEnabled={false}
          ItemSeparatorComponent={() => <View style={{ height: 12 }} />}
          ListEmptyComponent={!loading && name && dob ? <Text style={[styles.muted, { color: c.muted }]}>{t('history.noChats')}</Text> : null}
          renderItem={({ item }) => {
            const badge = emotionBadge(item.emotion);
            const stamp = formatStamp(item.created_at);
            return (
              <View style={[styles.card, { backgroundColor: c.card, borderColor: c.cardBorder }, shadowStyle(colorScheme)]}>
                <View style={styles.cardTop}>
                  {!!stamp && <Text style={[styles.stamp, { color: c.muted }]}>{stamp}</Text>}
                  {badge ? (
                    <View
                      style={[
                        styles.badge,
                        {
                          borderColor: `${badge.tint}55`,
                          backgroundColor: c.surfaceSubtle,
                        },
                      ]}
                    >
                      <Text style={[styles.badgeText, { color: badge.tint }]}>
                        {badge.emoji} {badge.label}
                      </Text>
                    </View>
                  ) : null}
                </View>
                <Text style={[styles.user, { color: c.text }]}>{t('history.youPrefix')} {item.message}</Text>
                <Text style={[styles.ai, { color: c.textSecondary }]}>{t('history.aiPrefix')} {item.reply}</Text>
              </View>
            );
          }}
        />

        <Text style={[styles.section, { color: c.primary }]}>{t('history.palmBeta')}</Text>
        <FlatList
          data={palms}
          keyExtractor={(item) => String(item.id)}
          scrollEnabled={false}
          ListEmptyComponent={
            !loading && name && dob ? (
              <View style={[styles.emptyCard, { backgroundColor: c.card, borderColor: c.cardBorder }, shadowStyle(colorScheme)]}>
                <Text style={[styles.emptyTitle, { color: c.text }]}>{t('history.scanPalmTitle')}</Text>
                <Text style={[styles.emptyBody, { color: c.muted }]}>
                  {t('history.scanPalmBody')}
                </Text>
                <Pressable
                  onPress={() => {
                    router.push('/camera');
                  }}
                  style={[styles.emptyBtn, { backgroundColor: c.primary }]}
                >
                  <Text style={styles.emptyBtnText}>{t('history.startPalmScan')}</Text>
                </Pressable>
              </View>
            ) : null
          }
          renderItem={({ item }) => (
            <View style={[styles.palmCard, { backgroundColor: c.card, borderColor: c.cardBorder }, shadowStyle(colorScheme)]}>
              <View style={styles.cardTop}>
                <Text style={[styles.palmText, { color: c.text }]}>{t('history.savedScan')}</Text>
                {!!item.created_at && <Text style={[styles.stamp, { color: c.muted }]}>{formatStamp(item.created_at)}</Text>}
              </View>
              {(() => {
                const localUri = String(item.local_uri ?? '').trim();
                if (localUri) return <Image source={{ uri: localUri }} style={styles.palmImage} />;

                const urlFromApi = item.image_url ?? '';
                const fallbackUrl = apiBaseUrl
                  ? `${apiBaseUrl}/${String(item.image_path ?? '').replace(/\\/g, '/').replace(/^\//, '')}`
                  : '';
                const uri = urlFromApi || fallbackUrl;
                return uri ? <Image source={{ uri }} style={styles.palmImage} /> : null;
              })()}

              {!!item.message && String(item.message).trim() ? (
                <Text style={[styles.palmMessage, { color: c.textSecondary }]}>{String(item.message).trim()}</Text>
              ) : null}
            </View>
          )}
        />
      </ScrollView>
    </ScreenWrapper>
  );
}

const styles = StyleSheet.create({
  title: { fontSize: 24, marginBottom: 8, fontWeight: 'bold' },
  subTitle: { marginBottom: 10 },
  section: { marginTop: 18, marginBottom: 10, fontWeight: '800', fontSize: 14 },

  statusPill: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 1,
    marginBottom: 12,
  },
  statusText: {
    fontWeight: '800',
    fontSize: 12,
    letterSpacing: 0.2,
  },

  card: {
    borderWidth: 1,
    padding: padding.card,
    borderRadius: radius.card,
  },
  cardTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 10,
    marginBottom: 8,
  },
  stamp: {
    fontSize: 12,
    fontWeight: '600',
  },
  badge: {
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  badgeText: {
    fontWeight: '800',
    fontSize: 12,
  },
  user: {},
  ai: { marginTop: 6 },
  emotion: { marginTop: 6 },
  palmRow: { paddingVertical: 6 },
  palmText: {},
  palmCard: {
    borderWidth: 1,
    padding: padding.card,
    borderRadius: radius.card,
  },
  snapshotCard: {
    borderWidth: 1,
    padding: padding.card,
    borderRadius: radius.card,
    gap: 10,
  },
  snapshotTitle: {
    fontWeight: '800',
    fontSize: 14,
  },
  snapshotRow: {
    gap: 4,
  },
  snapshotLabel: {
    fontWeight: '800',
    fontSize: 12,
    letterSpacing: 0.3,
  },
  snapshotValue: {
    fontSize: 14,
    lineHeight: 20,
  },
  emptyCard: {
    borderWidth: 1,
    padding: padding.card,
    borderRadius: radius.card,
  },
  emptyTitle: {
    fontWeight: '800',
    fontSize: 16,
    marginBottom: 8,
  },
  emptyBody: {
    lineHeight: 20,
  },
  emptyBtn: {
    marginTop: 12,
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
  },
  emptyBtnText: {
    color: 'white',
    fontWeight: '800',
  },
  palmImage: {
    width: '100%',
    height: 180,
    borderRadius: 12,
    marginTop: 10,
    borderWidth: 1,
  },
  palmMessage: {
    marginTop: 10,
    lineHeight: 20,
  },
  muted: {},
  error: { marginTop: 6 },
});
