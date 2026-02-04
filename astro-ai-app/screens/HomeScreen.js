import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  Share,
  StyleSheet,
  ScrollView,
  Platform,
  Animated,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import InsightCard from '../components/InsightCard';
import { useRouter } from 'expo-router';
import { api } from '../lib/api';
import { useUser } from '../context/user-context';
import { shadowStyle, useAppColors } from '../lib/ui-theme';
import { formatZodiacLabel, getZodiacFromDob } from '../lib/zodiac';
import { registerForPushNotificationsAsync } from '../utils/notifications';
import { useTranslation } from 'react-i18next';
import { useColorScheme } from '../hooks/use-color-scheme';
import ActionMenu from '../components/ui/ActionMenu';
import PressScale from '../components/ui/PressScale';
import { padding, radius } from '../lib/theme';

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function hashString(input) {
  const str = String(input ?? '');
  let h = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function getTodayKey() {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

function pick(list, seed) {
  if (!Array.isArray(list) || list.length === 0) return '';
  const idx = Math.abs(seed) % list.length;
  return list[idx];
}

function formatLocalTime(iso) {
  try {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  } catch {
    return '';
  }
}

export default function HomeScreen({ navigation } = {}) {
  const router = useRouter();
  const { user, language } = useUser();
  const c = useAppColors();
  const colorScheme = useColorScheme();
  const { t } = useTranslation();
  const [insight, setInsight] = useState('');
  const [zodiac, setZodiac] = useState('');
  const [pushToken, setPushToken] = useState(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);
  const insightFade = useRef(new Animated.Value(1)).current;

  const nav = useMemo(() => {
    if (navigation) return navigation;
    const routeMap = {
      Home: '/(tabs)',
      Chat: '/(tabs)/explore',
      Login: '/login',
      History: '/history',
      Palm: '/camera',
      Weekly: '/weekly',
    };
    return {
      navigate: (name) => router.push(routeMap[name] ?? '/(tabs)'),
      replace: (name) => router.replace(routeMap[name] ?? '/(tabs)'),
    };
  }, [navigation, router]);

  const fetchDailyInsight = async () => {
    setError('');
    setLoading(true);
    Animated.timing(insightFade, { toValue: 0, duration: 200, useNativeDriver: true }).start();
    try {
      const payload = {
        name: user?.name?.trim() || 'Friend',
        dob: user?.dob?.trim() || '2000-01-01',
        token: pushToken,
        language: language || 'en',
      };
      const res = await api.post('/daily-insight', payload);
      setInsight(res?.data?.insight ?? '');
      setZodiac(res?.data?.zodiac ?? getZodiacFromDob(payload.dob));
    } catch (_e) {
      setError(t('errors.backendOffline'));
      setInsight('');
      setZodiac(getZodiacFromDob(user?.dob?.trim() || ''));
    } finally {
      setLoading(false);
      Animated.timing(insightFade, { toValue: 1, duration: 200, useNativeDriver: true }).start();
    }
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const token = await registerForPushNotificationsAsync();
      if (!cancelled) setPushToken(token);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    fetchDailyInsight();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const todayKey = useMemo(() => getTodayKey(), []);
  const sign = useMemo(() => {
    const fromApi = (zodiac || '').trim();
    if (fromApi) return fromApi;
    return getZodiacFromDob(user?.dob?.trim() || '');
  }, [user?.dob, zodiac]);

  const seed = useMemo(() => {
    return hashString(`${user?.name || ''}|${user?.dob || ''}|${todayKey}|${sign}`);
  }, [sign, todayKey, user?.dob, user?.name]);

  const energy = useMemo(() => {
    // 55..95, stable for the day.
    return clamp(55 + (seed % 41), 45, 98);
  }, [seed]);

  const energyLabel = useMemo(() => {
    if (energy >= 80) return t('home.highMomentum');
    if (energy >= 65) return t('home.focusedSteady');
    return t('home.gentlePace');
  }, [energy, t]);

  const moodInfluence = useMemo(() => {
    return pick(
      ['Action‑oriented', 'Grounded', 'Social', 'Reflective', 'Confident', 'Precise', 'Balanced', 'Intense', 'Curious'],
      seed + 13
    );
  }, [seed]);

  const luckyFocus = useMemo(() => {
    return pick(['Career', 'Money', 'Love', 'Mindset', 'Health', 'Creativity', 'Social', 'Home'], seed + 31);
  }, [seed]);

  const focusLine = useMemo(() => {
    const prompts = [
      'One bold step moves everything. Avoid overthinking small risks.',
      'Pick one priority and finish it. Momentum beats perfection.',
      'Be honest about what you want, then choose the simplest next action.',
      'Protect your attention. Say “no” to one distraction today.',
      'Lead with calm confidence. Keep your tone clean and direct.',
    ];
    return pick(prompts, seed + 97);
  }, [seed]);

  const luckyColor = useMemo(() => pick(['Blue', 'Indigo', 'Emerald', 'Rose', 'Gold', 'Silver', 'Violet', 'Teal'], seed + 7), [seed]);
  const luckyNumber = useMemo(() => 1 + (Math.abs(seed + 3) % 9), [seed]);
  const powerHour = useMemo(() => pick(['7–9 AM', '11 AM–1 PM', '3–5 PM', '6–8 PM', '9–10 PM'], seed + 19), [seed]);

  const zodiacLabel = useMemo(() => formatZodiacLabel(sign), [sign]);

  const shareMessage = useMemo(() => {
    const header = `AstroKing • Daily Guidance\n${todayKey}${zodiacLabel ? ` • ${zodiacLabel}` : ''}`;
    const energyLine = `Today’s Energy: ${energy}% (${energyLabel})`;
    const moodLine = `Mood influence: ${moodInfluence}`;
    const focusLineText = `Lucky focus: ${luckyFocus}`;
    const luckyLine = `Lucky elements: ${luckyColor} • ${luckyNumber} • ${powerHour}`;
    const insightLine = insight ? `Daily insight: ${insight}` : '';
    const focus = `Your focus today: ${focusLine}`;
    return [header, '', energyLine, moodLine, focusLineText, luckyLine, '', focus, insightLine ? `\n${insightLine}` : '']
      .join('\n')
      .trim();
  }, [energy, energyLabel, focusLine, insight, luckyColor, luckyFocus, luckyNumber, moodInfluence, powerHour, todayKey, zodiacLabel]);

  const onShare = async () => {
    try {
      await Share.share({
        title: 'AstroKing — Daily Guidance',
        message: shareMessage,
      });
    } catch {
      // ignore
    }
  };

  return (
    <SafeAreaView style={[styles.safe, { backgroundColor: c.bg }]}>
      <ScrollView
        contentContainerStyle={[styles.container, { backgroundColor: c.bg }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.header}>
          <View style={styles.headerTopRow}>
            <View style={styles.headerText}>
              <Text style={[styles.kicker, { color: c.muted }]}>AstroKing</Text>
              <Text style={[styles.title, { color: c.text }]}>{t('home.dailyGuidance')}</Text>
              <Text style={[styles.subtitle, { color: c.muted }]}>
                {user?.name ? t('home.welcomeBack', { name: user.name }) : t('home.setProfileForPersonalized')}
              </Text>
              {!!zodiacLabel && (
                <Text style={[styles.zodiac, { color: c.text }]}>
                  {zodiacLabel}
                  <Text style={{ color: c.muted }}>{`  •  ${todayKey}`}</Text>
                </Text>
              )}
            </View>

            <View style={styles.headerActions}>
              <PressScale
                accessibilityLabel={t('home.refreshInsight')}
                onPress={fetchDailyInsight}
                disabled={loading}
                style={[
                  styles.smallButton,
                  { backgroundColor: c.card, borderColor: c.cardBorder },
                  loading && { opacity: 0.6 },
                ]}
              >
                <Text style={[styles.smallButtonText, { color: c.text }]}>
                  {loading ? t('common.refreshing') : t('home.refreshInsight')}
                </Text>
              </PressScale>

              <PressScale
                accessibilityLabel="More"
                onPress={() => setMenuOpen(true)}
                style={[styles.ellipsisButton, { backgroundColor: c.card, borderColor: c.cardBorder }]}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Text style={[styles.ellipsisText, { color: c.text }]}>⋯</Text>
              </PressScale>
            </View>
          </View>
        </View>

        <View
          style={[
            styles.energyCard,
            { backgroundColor: c.card, borderColor: c.cardBorder },
            shadowStyle(colorScheme),
          ]}
        >
          <View style={styles.energyHeader}>
            <Text style={[styles.energyTitle, { color: c.text }]}>{t('home.todaysEnergy')}</Text>
            <Text style={[styles.energyMeta, { color: c.muted }]}>
              {energy}% · {energyLabel}
            </Text>
          </View>
          <View
            style={[
              styles.energyBar,
              { backgroundColor: c.pressBg },
            ]}
          >
            <View style={[styles.energyFill, { width: `${energy}%`, backgroundColor: c.primary }]} />
          </View>
          <View style={styles.energyRow}>
            <Text style={[styles.energyPill, { color: c.text, backgroundColor: c.bg2, borderColor: c.cardBorder }]}>
              🔥 {moodInfluence}
            </Text>
            <Text style={[styles.energyPill, { color: c.text, backgroundColor: c.bg2, borderColor: c.cardBorder }]}>
              🎯 {luckyFocus}
            </Text>
          </View>
        </View>

        <View style={styles.cards}>
          <Animated.View style={{ opacity: insightFade }}>
            <InsightCard
              title={t('nav.dailyInsight')}
              body={error ? error : insight || `${t('weekly.noInsightYet')} ${t('home.refreshInsight')}.`}
              footer={Platform.OS === 'web' ? 'Tip: Use /login to update your profile.' : undefined}
            />
          </Animated.View>
          <InsightCard
            title="Your Focus Today"
            body={focusLine}
            footer={`Power hour: ${powerHour}`}
          />
          <InsightCard
            title="Lucky Elements"
            body={`🎨 Lucky color: ${luckyColor}\n🔢 Lucky number: ${luckyNumber}\n⏰ Power hour: ${powerHour}`}
          />
          <InsightCard
            title="Astro Tip"
            body={
              sign
                ? `For ${sign}: choose one decision and make it clean. If you feel scattered, take a 3‑minute reset.`
                : 'Choose one decision and make it clean. If you feel scattered, take a 3‑minute reset.'
            }
            footer={formatLocalTime(new Date().toISOString()) ? `Updated ${formatLocalTime(new Date().toISOString())}` : undefined}
          />
        </View>

        <PressScale
          onPress={() => nav.navigate?.('Chat')}
          style={[styles.primaryButton, { backgroundColor: c.primary }]}
        >
          <Text style={styles.primaryButtonText}>{t('nav.chat')}</Text>
        </PressScale>

        <ActionMenu
          visible={menuOpen}
          onClose={() => setMenuOpen(false)}
          title={t('home.dailyGuidance')}
          items={[
            {
              key: 'profile',
              label: user?.name ? t('home.editProfile') : t('home.setProfile'),
              onPress: () => nav.navigate?.('Login'),
            },
            { key: 'share', label: t('home.share'), onPress: onShare },
            { key: 'palm', label: `${t('nav.palmScan')} (Beta)`, onPress: () => nav.navigate?.('Palm') },
            { key: 'history', label: t('nav.yourHistory'), onPress: () => nav.navigate?.('History') },
            { key: 'weekly', label: t('nav.weeklyReport'), onPress: () => nav.navigate?.('Weekly') },
          ]}
        />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
  },
  container: {
    padding: 20,
    gap: 14,
    paddingBottom: 28,
  },
  header: {
    marginTop: 6,
    marginBottom: 2,
  },
  headerTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  headerText: {
    gap: 6,
    flexShrink: 1,
  },
  headerActions: {
    alignItems: 'flex-end',
    gap: 10,
  },
  kicker: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
  },
  subtitle: {
    marginTop: 2,
  },
  zodiac: {
    marginTop: 6,
    fontWeight: '800',
  },
  cards: {
    gap: 12,
    marginTop: 4,
  },
  smallButton: {
    borderWidth: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 999,
  },
  smallButtonText: {
    fontWeight: '800',
  },
  ellipsisButton: {
    width: 42,
    height: 42,
    borderWidth: 1,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
  },
  ellipsisText: {
    fontSize: 22,
    fontWeight: '900',
    marginTop: -6,
  },
  primaryButton: {
    marginTop: 10,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: 'white',
    fontWeight: '700',
    fontSize: 16,
  },
  energyCard: {
    borderWidth: 1,
    borderRadius: radius.card,
    padding: padding.card,
    gap: 10,
  },
  energyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    gap: 10,
  },
  energyTitle: {
    fontSize: 18,
    fontWeight: '900',
  },
  energyMeta: {
    fontWeight: '800',
  },
  energyBar: {
    height: 10,
    borderRadius: 999,
    overflow: 'hidden',
  },
  energyFill: {
    height: 10,
    borderRadius: 999,
  },
  energyRow: {
    flexDirection: 'row',
    gap: 10,
    flexWrap: 'wrap',
  },
  energyPill: {
    borderWidth: 1,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 999,
    fontWeight: '800',
    overflow: 'hidden',
  },
});
