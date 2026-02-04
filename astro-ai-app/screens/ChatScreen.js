import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  TextInput,
  Pressable,
  Share,
  StyleSheet,
  FlatList,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { api } from '../lib/api';
import { useUser } from '../context/user-context';
import { useAppColors, shadowStyle } from '../lib/ui-theme';
import SessionSummaryCard from '../components/SessionSummaryCard';
import { useTranslation } from 'react-i18next';
import ActionMenu from '../components/ui/ActionMenu';
import PressScale from '../components/ui/PressScale';
import FadeIn from '../components/ui/FadeIn';
import { LinearGradient } from 'expo-linear-gradient';
import { useColorScheme } from '../hooks/use-color-scheme';
import { accentGlowStyle, padding, radius } from '../lib/theme';
import { detectIntent, intentEmoji } from '../lib/ux/intent';
import { getHint } from '../lib/ux/hints';
import { getRecentTopics, updateRecentTopics } from '../lib/ux/memory';
import { getToneHint, saveFeedback } from '../lib/ux/feedback';

export default function ChatScreen() {
  const router = useRouter();
  const { user, language } = useUser();
  const c = useAppColors();
  const colorScheme = useColorScheme();
  const { t } = useTranslation();
  const aiTextColor = c.isDark ? c.textSecondary : c.text;
  const listRef = useRef(null);
  const inputRef = useRef(null);
  const restoreTimerRef = useRef(null);

  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      id: '1',
      role: 'assistant',
      text: "I'm tuned into your energy today. Let's keep this simple and actionable.",
    },
  ]);
  const [sending, setSending] = useState(false);
  const [connectionIssue, setConnectionIssue] = useState(false);
  const [connectionRestored, setConnectionRestored] = useState(false);
  const [lastFailedPayload, setLastFailedPayload] = useState(null);
  const [memoryContext, setMemoryContext] = useState('');
  const [thinkingDots, setThinkingDots] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);

  const [liveIntent, setLiveIntent] = useState('general');
  const [liveHint, setLiveHint] = useState('');
  const [recentTopics, setRecentTopics] = useState([]);
  const [toneHint, setToneHint] = useState('balanced');

  const [showSummary, setShowSummary] = useState(false);
  const [generatedSummary, setGeneratedSummary] = useState(null);
  const [savingSummary, setSavingSummary] = useState(false);
  const [sessionMessageCount, setSessionMessageCount] = useState(0);
  const [streakTriggeredDay, setStreakTriggeredDay] = useState('');

  const userKey = useMemo(() => {
    const name = user?.name || '';
    const dob = user?.dob || '';
    return `${name}|${dob}`;
  }, [user?.dob, user?.name]);

  const data = useMemo(() => messages, [messages]);

  const lastAssistantText = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const m = messages[i];
      if (m?.role === 'assistant' && typeof m?.text === 'string' && m.text.trim()) return m.text.trim();
    }
    return '';
  }, [messages]);

  const onShare = async () => {
    const name = user?.name?.trim() || '';
    const message = lastAssistantText
      ? `AstroKing • Chat Insight${name ? ` for ${name}` : ''}\n\n${lastAssistantText}`
      : 'AstroKing • Chat Insight\n\nChat with the assistant to generate an insight to share.';

    try {
      await Share.share({
        title: 'AstroKing — Chat Insight',
        message,
      });
    } catch {
      // ignore
    }
  };

  const shareBubble = async (text) => {
    const t = String(text ?? '').trim();
    if (!t) return;
    const name = user?.name?.trim() || '';
    try {
      await Share.share({
        title: 'AstroKing — Chat Insight',
        message: `AstroKing • Chat Insight${name ? ` for ${name}` : ''}\n\n${t}`,
      });
    } catch {
      // ignore
    }
  };

  const detectContext = (text) => {
    const t = String(text || '').toLowerCase();
    if (t.match(/exam|exams|test|tests|school|college|study/)) return 'your exams';
    if (t.match(/interview|promotion|resume|portfolio|boss|job/)) return 'your career';
    if (t.match(/breakup|relationship|dating|partner|crush|love/)) return 'your relationships';
    if (t.match(/stress|anxiety|overthink|panic|burnout/)) return 'your stress levels';
    if (t.match(/sleep|energy|gym|health|diet/)) return 'your health and energy';
    return '';
  };

  useEffect(() => {
    return () => {
      if (restoreTimerRef.current) clearTimeout(restoreTimerRef.current);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [topics, tone] = await Promise.all([getRecentTopics(userKey), getToneHint(userKey)]);
        if (cancelled) return;
        setRecentTopics(Array.isArray(topics) ? topics : []);
        setToneHint(tone || 'balanced');
      } catch {
        if (cancelled) return;
        setRecentTopics([]);
        setToneHint('balanced');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [userKey]);

  const retryLast = async () => {
    if (!lastFailedPayload) return;
    setConnectionIssue(false);
    setConnectionRestored(false);
    setSending(true);
    try {
      const res = await api.post('/chat', { ...lastFailedPayload, language: language || 'en' });
      const replyText = res?.data?.reply ?? 'No reply received.';
      setMessages((prev) => [...prev, { id: String(Date.now() + 1), role: 'assistant', text: replyText }]);
      setLastFailedPayload(null);

      setConnectionRestored(true);
      if (restoreTimerRef.current) clearTimeout(restoreTimerRef.current);
      restoreTimerRef.current = setTimeout(() => setConnectionRestored(false), 2600);
    } catch (e) {
      // behind the scenes only
      console.log('Server not reachable:', e);
      setConnectionIssue(true);
    } finally {
      setSending(false);
      requestAnimationFrame(() => listRef.current?.scrollToEnd?.({ animated: true }));
    }
  };

  const quickActions = useMemo(
    () => [
      { key: 'money', label: t('chat.quick.moneyLabel'), prompt: t('chat.quick.moneyPrompt') },
      { key: 'love', label: t('chat.quick.loveLabel'), prompt: t('chat.quick.lovePrompt') },
      { key: 'mind', label: t('chat.quick.mindLabel'), prompt: t('chat.quick.mindPrompt') },
      { key: 'career', label: t('chat.quick.careerLabel'), prompt: t('chat.quick.careerPrompt') },
      {
        key: 'decision',
        label: t('chat.quick.decisionLabel'),
        prompt: t('chat.quick.decisionPrompt'),
      },
    ],
    [t]
  );

  const generateSummary = useCallback(async () => {
    const userMessages = messages.filter((m) => m.role === 'user').map((m) => m.text);
    const assistantMessages = messages.filter((m) => m.role === 'assistant').map((m) => m.text);

    const text = userMessages.join(' ').toLowerCase();
    let focus = '✨ General';
    if (text.match(/money|finance|finances|debt|invest|budget/)) focus = '💰 Money & Security';
    else if (text.match(/job|work|career|salary|interview|boss/)) focus = '🎯 Career & Growth';
    else if (text.match(/love|relationship|partner|dating|crush/)) focus = '❤️ Love & Relationships';
    else if (text.match(/stress|anxiety|mind|mental|worry|overthink/)) focus = '🧠 Mental Clarity';
    else if (text.match(/health|body|energy|sleep|gym/)) focus = '💪 Health & Energy';

    let keyInsight = 'A small improvement to your routine will compound fast.';
    const growthWords = ['improve', 'focus', 'build', 'step', 'growth', 'forward', 'progress', 'small', 'compound', 'routine'];
    for (let i = assistantMessages.length - 1; i >= Math.max(0, assistantMessages.length - 10); i -= 1) {
      const msg = assistantMessages[i] || '';
      if (growthWords.some((w) => msg.toLowerCase().includes(w))) {
        keyInsight = msg.length > 150 ? msg.slice(0, 147) + '...' : msg;
        break;
      }
    }

    const bucket = (emotion) => {
      const e = String(emotion || '').toLowerCase();
      if (e.includes('happy')) return 'positive';
      if (e.includes('sad') || e.includes('anxiety') || e.includes('stress')) return 'negative';
      return 'neutral';
    };

    const emoji = (b) => {
      if (b === 'positive') return '😊';
      if (b === 'negative') return '😰';
      return '😐';
    };

    const trendLabel = (from, to) => {
      if (from === 'neutral' && to === 'positive') return 'Improving Stability';
      if (from === 'negative' && to === 'neutral') return 'Recovering Balance';
      if (from === 'negative' && to === 'positive') return 'Major Lift';
      if (from === 'positive' && to === 'neutral') return 'Settling + Integrating';
      if (from === 'neutral' && to === 'negative') return 'Under Pressure';
      if (from === 'positive' && to === 'negative') return 'Overextended';
      return 'Steady';
    };

    let moodTrend = '😐 → 😐 Steady';
    let phase = 'stabilizing';
    try {
      if (user?.name && user?.dob) {
        const res = await api.get(
          `/history/chat?name=${encodeURIComponent(user.name)}&dob=${encodeURIComponent(user.dob)}&limit=40`
        );
        const items = Array.isArray(res?.data?.items) ? res.data.items.slice().reverse() : [];
        const moods = items.map((r) => r?.emotion).filter(Boolean).map(bucket);
        if (moods.length >= 2) {
          const from = moods[0];
          const to = moods[moods.length - 1];
          moodTrend = `${emoji(from)} → ${emoji(to)} ${trendLabel(from, to)}`;
          phase = to === 'positive' ? 'growth' : to === 'negative' ? 'reset' : 'stabilizing';
        }
      }
    } catch {
      // ignore
    }

    const reflectionTemplates = {
      growth: [
        "You're currently in a building phase - progress may feel small but it's compounding.",
        "You're in a growth phase - keep your routine simple and consistent.",
      ],
      stabilizing: [
        "You're stabilizing - progress is quiet but real.",
        "You're integrating lessons - steady steps are your superpower right now.",
      ],
      reset: [
        "You're in a reset phase - protect your energy and return to basics.",
        "You're recalibrating - one clean habit will restart your momentum.",
      ],
    };
    const pool = reflectionTemplates[phase] || reflectionTemplates.stabilizing;
    const reflection = pool[Math.floor(Math.random() * pool.length)];

    const nextStepByFocus = {
      '💰 Money & Security': 'Next step: choose one money action (budget, apply, negotiate) and do it today.',
      '🎯 Career & Growth': 'Next step: do one compounding career action (skill, outreach, portfolio) today.',
      '❤️ Love & Relationships': 'Next step: have one honest conversation - clarity creates closeness.',
      '🧠 Mental Clarity': 'Next step: choose one calming routine (walk, breath, journal) and repeat it for 7 days.',
      '💪 Health & Energy': 'Next step: lock one health habit (sleep window, hydration, movement) for this week.',
      '✨ General': 'Next step: focus on one consistent habit this week.',
    };
    const nextStep = nextStepByFocus[focus] || nextStepByFocus['✨ General'];

    setGeneratedSummary({
      focus,
      key_insight: keyInsight,
      mood_trend: moodTrend,
      reflection,
      next_step: nextStep,
    });
    setShowSummary(true);
  }, [messages, user?.dob, user?.name]);

  useEffect(() => {
    // Trigger: 5+ messages in a session
    if (sessionMessageCount >= 5 && !showSummary && !generatedSummary) {
      void generateSummary();
    }
  }, [generateSummary, generatedSummary, sessionMessageCount, showSummary]);

  useEffect(() => {
    // Trigger: 3-day chat streak (after at least one message in this session)
    if (sessionMessageCount < 1) return;
    if (!user?.name || !user?.dob) return;
    if (showSummary || generatedSummary) return;

    const today = new Date().toISOString().slice(0, 10);
    if (streakTriggeredDay === today) return;

    (async () => {
      try {
        const res = await api.get(
          `/history/chat?name=${encodeURIComponent(user.name)}&dob=${encodeURIComponent(user.dob)}&limit=120`
        );
        const items = Array.isArray(res?.data?.items) ? res.data.items : [];
        const dayKeys = new Set(
          items
            .map((r) => String(r?.created_at || '').slice(0, 10))
            .filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d))
        );

        let streak = 0;
        const cursor = new Date(today);
        for (let i = 0; i < 14; i += 1) {
          const key = cursor.toISOString().slice(0, 10);
          if (!dayKeys.has(key)) break;
          streak += 1;
          cursor.setDate(cursor.getDate() - 1);
        }

        if (streak >= 3) {
          setStreakTriggeredDay(today);
          void generateSummary();
        }
      } catch {
        // ignore
      }
    })();
  }, [generateSummary, generatedSummary, sessionMessageCount, showSummary, streakTriggeredDay, user?.dob, user?.name]);

  const send = async (overrideText) => {
    const trimmed = String(overrideText ?? input).trim();
    if (!trimmed) return;

    // Requirement: hide keyboard first, then send.
    inputRef.current?.blur?.();
    Keyboard.dismiss();

    setConnectionIssue(false);

    if (!user?.name || !user?.dob) {
      setMessages((prev) => [
        ...prev,
        {
          id: String(Date.now()),
          role: 'assistant',
          text: t('chat.profileFirst'),
        },
      ]);
      router.push('/login');
      return;
    }

    const userMessage = { id: String(Date.now()), role: 'user', text: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setSending(true);

    const intent = detectIntent(trimmed);
    setLiveIntent(intent);
    setLiveHint('');

    let nextTopics = recentTopics;
    try {
      nextTopics = await updateRecentTopics(userKey, intent);
      setRecentTopics(Array.isArray(nextTopics) ? nextTopics : []);
    } catch {
      // ignore
    }

    let tone = toneHint;
    try {
      tone = await getToneHint(userKey);
      setToneHint(tone || 'balanced');
    } catch {
      tone = toneHint || 'balanced';
    }

    const ctx = detectContext(trimmed) || memoryContext;
    if (ctx) setMemoryContext(ctx);

    try {
      const payload = {
        name: user.name,
        dob: user.dob,
        message: trimmed,
        context: ctx || undefined,
        language: language || 'en',
        intent,
        recent_topics: nextTopics,
        tone,
      };
      const res = await api.post('/chat', payload);
      const replyText = res?.data?.reply ?? 'No reply received.';
      setMessages((prev) => [...prev, { id: String(Date.now() + 1), role: 'assistant', text: replyText }]);
    } catch (e) {
      // behind the scenes only
      console.log('Server not reachable:', e);
      setConnectionIssue(true);
      setLastFailedPayload({
        name: user.name,
        dob: user.dob,
        message: trimmed,
        context: ctx || undefined,
        language: language || 'en',
        intent,
        recent_topics: nextTopics,
        tone,
      });
    } finally {
      setSending(false);
      setSessionMessageCount((prev) => prev + 1);
      requestAnimationFrame(() => listRef.current?.scrollToEnd?.({ animated: true }));
    }
  };

  const onComposerChangeText = (text) => {
    setInput(text);
    const intent = detectIntent(text);
    setLiveIntent(intent);
    const hint = getHint(text, intent, t);
    setLiveHint(hint || '');
  };

  useEffect(() => {
    if (!sending) {
      setThinkingDots('');
      return;
    }
    let tick = 0;
    const id = setInterval(() => {
      tick = (tick + 1) % 4;
      setThinkingDots('.'.repeat(tick));
    }, 350);
    return () => clearInterval(id);
  }, [sending]);

  const saveSummary = async () => {
    if (!generatedSummary || !user?.name || !user?.dob) return;
    setSavingSummary(true);
    try {
      await api.post('/session-summary', {
        name: user.name,
        dob: user.dob,
        summary: generatedSummary,
      });
      setShowSummary(false);
      setGeneratedSummary(null);
      setSessionMessageCount(0);
    } catch {
      // ignore
    } finally {
      setSavingSummary(false);
    }
  };

  const closeSummary = () => {
    setShowSummary(false);
    setGeneratedSummary(null);
    setSessionMessageCount(0);
  };
  return (
    <SafeAreaView style={[styles.safe, { backgroundColor: c.bg }]}>
      <KeyboardAvoidingView
        style={[styles.container, { backgroundColor: c.bg }]}
        behavior={Platform.OS === 'ios' ? 'padding' : 'padding'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 6 : 0}
      >
        <View style={styles.header}>
          <View style={styles.headerText}>
            <Text style={[styles.title, { color: c.text }]}>{t('chat.title')}</Text>
            <Text style={[styles.subtitle, { color: c.muted }]}>{t('chat.subtitle')}</Text>
          </View>
          <View style={styles.headerActions}>
            <PressScale
              onPress={generateSummary}
              disabled={messages.length < 3 || sending}
              style={[
                styles.primaryHeaderButton,
                { backgroundColor: c.card, borderColor: c.cardBorder },
                (messages.length < 3 || sending) && { opacity: 0.6 },
              ]}
              accessibilityLabel={t('chat.endSession')}
            >
              <Text style={[styles.profileButtonText, { color: c.text }]}>{t('chat.endSession')}</Text>
            </PressScale>

            <PressScale
              onPress={() => setMenuOpen(true)}
              style={[styles.ellipsisButton, { backgroundColor: c.card, borderColor: c.cardBorder }]}
              hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              accessibilityLabel="More"
            >
              <Text style={[styles.ellipsisText, { color: c.text }]}>⋯</Text>
            </PressScale>
          </View>
        </View>

        <ActionMenu
          visible={menuOpen}
          onClose={() => setMenuOpen(false)}
          title={t('chat.title')}
          items={[
            {
              key: 'share',
              label: t('chat.share'),
              onPress: onShare,
              disabled: !lastAssistantText || sending,
            },
            {
              key: 'profile',
              label: user?.name ? t('chat.profile') : t('chat.setProfile'),
              onPress: () => router.push('/login'),
            },
          ]}
        />

        <FlatList
          ref={listRef}
          data={data}
          keyExtractor={(item) => item.id}
          style={styles.listView}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
          onContentSizeChange={() => listRef.current?.scrollToEnd?.({ animated: true })}
          ListHeaderComponent={
            <View style={styles.quickWrap}>
              <Text style={[styles.quickTitle, { color: c.muted }]}>{t('chat.quickStarts')}</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.quickRow}>
                {quickActions.map((a) => (
                  (() => {
                    const key = a.key;
                    const tint =
                      key === 'money'
                        ? 'rgba(245,158,11,0.28)'
                        : key === 'love'
                          ? 'rgba(244,63,94,0.22)'
                          : key === 'mind'
                            ? 'rgba(56,189,248,0.20)'
                            : key === 'career'
                              ? 'rgba(34,197,94,0.20)'
                              : 'rgba(168,85,247,0.18)';
                    const border =
                      key === 'money'
                        ? 'rgba(245,158,11,0.55)'
                        : key === 'love'
                          ? 'rgba(244,63,94,0.45)'
                          : key === 'mind'
                            ? 'rgba(56,189,248,0.40)'
                            : key === 'career'
                              ? 'rgba(34,197,94,0.38)'
                              : 'rgba(168,85,247,0.35)';
                    return (
                      <PressScale
                        key={a.key}
                        onPress={() => send(a.prompt)}
                        disabled={sending}
                        androidRippleColor={c.pressBg}
                        style={[
                          styles.quickChip,
                          { backgroundColor: c.card, borderColor: c.cardBorder },
                          shadowStyle(colorScheme),
                          sending && { opacity: 0.6 },
                        ]}
                      >
                        <View style={[styles.quickChipInner, { backgroundColor: tint, borderColor: border }]}>
                          <Text style={[styles.quickChipText, { color: c.text }]}>{a.label}</Text>
                        </View>
                      </PressScale>
                    );
                  })()
                ))}
              </ScrollView>
            </View>
          }
          renderItem={({ item }) => {
            const isUser = item.role === 'user';
            return (
              <FadeIn>
                <Pressable
                  onLongPress={!isUser ? () => shareBubble(item.text) : undefined}
                  delayLongPress={220}
                  style={[styles.bubble, isUser ? styles.userBubble : styles.assistantBubble, shadowStyle(colorScheme)]}
                >
                  {isUser ? (
                    c.isDark ? (
                      <LinearGradient
                        colors={[c.primary, '#3D4DFF']}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                        style={styles.userBubbleInner}
                      >
                        <Text style={[styles.bubbleText, { color: 'white' }]}>{item.text}</Text>
                      </LinearGradient>
                    ) : (
                      <View style={[styles.userBubbleInner, { backgroundColor: c.primary }]}>
                        <Text style={[styles.bubbleText, { color: 'white' }]}>{item.text}</Text>
                      </View>
                    )
                  ) : (
                    <View style={[styles.assistantBubbleInner, { backgroundColor: c.card, borderColor: c.cardBorder }]}>
                      <Text style={[styles.bubbleText, { color: aiTextColor }]}>{item.text}</Text>
                    </View>
                  )}
                </Pressable>
              </FadeIn>
            );
          }}
          ListFooterComponent={
            sending ? (
              <View style={[styles.bubble, styles.assistantBubble, shadowStyle(colorScheme)]}>
                <View style={[styles.assistantBubbleInner, { backgroundColor: c.card, borderColor: c.cardBorder }]}>
                  <Text style={[styles.bubbleText, { color: c.muted }]}>
                    {t('chat.thinking')}
                    {thinkingDots}{' '}
                  </Text>
                </View>
              </View>
            ) : null
          }
        />

        {connectionIssue ? (
          <View
            style={[
              styles.connectionCard,
              shadowStyle(colorScheme),
              {
                backgroundColor: c.isDark ? 'rgba(239,68,68,0.14)' : 'rgba(239,68,68,0.08)',
                borderColor: c.isDark ? 'rgba(239,68,68,0.34)' : 'rgba(239,68,68,0.22)',
              },
            ]}
          >
            <Text style={[styles.connectionTitle, { color: c.danger }]}>{t('chat.connectionTitle')}</Text>
            <Text style={[styles.connectionBody, { color: c.muted }]}>{t('chat.connectionBody')}</Text>
            <View style={styles.connectionActions}>
              <Pressable
                onPress={retryLast}
                disabled={sending || !lastFailedPayload}
                style={[
                  styles.retryBtn,
                  {
                    borderColor: c.primary,
                    backgroundColor: 'transparent',
                  },
                  (sending || !lastFailedPayload) && { opacity: 0.55 },
                ]}
              >
                <Text style={[styles.retryText, { color: c.primary }]}>{t('chat.retry')}</Text>
              </Pressable>
            </View>
          </View>
        ) : connectionRestored ? (
          <View
            style={[
              styles.connectionCard,
              shadowStyle(colorScheme),
              {
                backgroundColor: c.isDark ? 'rgba(34,197,94,0.14)' : 'rgba(34,197,94,0.08)',
                borderColor: c.isDark ? 'rgba(34,197,94,0.34)' : 'rgba(34,197,94,0.22)',
              },
            ]}
          >
            <Text style={[styles.connectionTitle, { color: '#22c55e' }]}>{t('chat.connectionRestoredTitle')}</Text>
            <Text style={[styles.connectionBody, { color: c.muted }]}>{t('chat.connectionRestoredBody')}</Text>
          </View>
        ) : null}

        <View style={[styles.composer, { borderTopColor: c.border, backgroundColor: c.bg }]}>
          {!!input.trim() && (liveIntent !== 'general' || !!liveHint) ? (
            <View style={styles.intentWrap}>
              <View style={styles.intentRow}>
                {liveIntent !== 'general' ? (
                  <View
                    style={[
                      styles.intentBadge,
                      {
                        backgroundColor: c.isDark ? 'rgba(59,130,246,0.18)' : 'rgba(59,130,246,0.10)',
                        borderColor: c.isDark ? 'rgba(59,130,246,0.35)' : 'rgba(59,130,246,0.22)',
                      },
                    ]}
                  >
                    <Text style={[styles.intentBadgeText, { color: c.text }]}>
                      {intentEmoji(liveIntent)} {t(`chat.intent.${liveIntent}`)}
                    </Text>
                  </View>
                ) : null}

                {toneHint && toneHint !== 'balanced' ? (
                  <View style={[styles.tonePill, { borderColor: c.border, backgroundColor: c.card }]}>
                    <Text style={[styles.tonePillText, { color: c.muted }]}>
                      {toneHint === 'supportive' ? '🤍' : '⚡'} {toneHint}
                    </Text>
                  </View>
                ) : null}
              </View>

              {!!liveHint ? <Text style={[styles.hintText, { color: c.muted }]}>{liveHint}</Text> : null}
            </View>
          ) : null}

          <View style={styles.composerRow}>
            <View
              style={[
                styles.inputWrap,
                { backgroundColor: c.inputBg, borderColor: c.inputBorder },
                Platform.OS === 'web' && {
                  boxShadow: c.isDark
                    ? 'inset 0px 1px 1px rgba(0,0,0,0.55)'
                    : 'inset 0px 1px 1px rgba(2,6,23,0.08)',
                },
              ]}
            >
              <TextInput
                ref={inputRef}
                value={input}
                onChangeText={onComposerChangeText}
                placeholder={t('chat.placeholder')}
                placeholderTextColor={c.placeholderSoft}
                style={[styles.input, { color: c.text }]}
                multiline
                returnKeyType="send"
                blurOnSubmit
                onSubmitEditing={() => send()}
              />
            </View>
            <PressScale
              onPress={() => send()}
              disabled={sending}
              style={[
                styles.send,
                { backgroundColor: c.primary },
                accentGlowStyle(colorScheme, c.primary),
                sending && { opacity: 0.7 },
              ]}
              androidRippleColor={c.pressBg}
            >
              <Text style={styles.sendText}>{sending ? '…' : t('chat.send')}</Text>
            </PressScale>
          </View>
        </View>
      </KeyboardAvoidingView>
      {showSummary && generatedSummary && (
        <SessionSummaryCard
          summary={generatedSummary}
          userName={user?.name}
          onSave={saveSummary}
          onClose={closeSummary}
          saving={savingSummary}
          variant="highlight"
          onFeedback={async (value) => {
            try {
              await saveFeedback(userKey, value);
              const nextTone = await getToneHint(userKey);
              setToneHint(nextTone || 'balanced');
            } catch {
              // ignore
            }
          }}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
  },
  container: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 10,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    marginBottom: 10,
  },
  headerText: {
    flex: 1,
    gap: 3,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
  },
  subtitle: {
    marginBottom: 0,
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  primaryHeaderButton: {
    borderWidth: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 999,
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
  profileButton: {
    borderWidth: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 999,
  },
  profileButtonText: {
    fontWeight: '700',
  },
  list: {
    paddingVertical: 10,
    gap: 12,
  },
  listView: {
    flex: 1,
  },
  quickWrap: {
    marginBottom: 6,
    gap: 8,
  },
  quickTitle: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
  },
  quickRow: {
    gap: 10,
    paddingRight: 8,
  },
  quickChip: {
    borderWidth: 1,
    borderRadius: radius.pill,
    overflow: 'hidden',
  },
  quickChipInner: {
    borderWidth: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: radius.pill,
  },
  quickChipText: {
    fontWeight: '800',
  },
  bubble: {
    maxWidth: '80%',
    borderRadius: radius.bubble,
    overflow: 'hidden',
  },
  userBubble: {
    alignSelf: 'flex-end',
  },
  assistantBubble: {
    alignSelf: 'flex-start',
  },
  userBubbleInner: {
    paddingVertical: padding.bubble,
    paddingHorizontal: padding.bubble,
    borderRadius: radius.bubble,
  },
  assistantBubbleInner: {
    borderWidth: 1,
    paddingVertical: padding.bubble,
    paddingHorizontal: padding.bubble,
    borderRadius: radius.bubble,
  },
  bubbleText: {
    lineHeight: 20,
  },
  composer: {
    gap: 10,
    paddingTop: 10,
    paddingBottom: 12,
    borderTopWidth: 1,
    alignItems: 'stretch',
  },
  composerRow: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'flex-end',
  },
  intentWrap: {
    gap: 8,
  },
  intentRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    alignItems: 'center',
  },
  intentBadge: {
    alignSelf: 'flex-start',
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 999,
    borderWidth: 1,
  },
  intentBadgeText: {
    fontSize: 12,
    fontWeight: '700',
  },
  tonePill: {
    alignSelf: 'flex-start',
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 999,
    borderWidth: 1,
  },
  tonePillText: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  hintText: {
    fontSize: 12,
    lineHeight: 16,
  },
  inputWrap: {
    flex: 1,
    borderWidth: 1,
    borderRadius: radius.input,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  input: {
    fontSize: 16,
    minHeight: 22,
    maxHeight: 110,
  },
  send: {
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: radius.input,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendText: {
    color: 'white',
    fontWeight: '700',
  },
  connectionCard: {
    marginHorizontal: 16,
    marginTop: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderRadius: radius.card,
    padding: padding.card,
  },
  connectionTitle: {
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 6,
  },
  connectionBody: {
    fontSize: 13,
    lineHeight: 18,
  },
  connectionActions: {
    marginTop: 10,
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  retryBtn: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 12,
    borderWidth: 1,
  },
  retryText: {
    fontWeight: '700',
    letterSpacing: 0.2,
  },
});
