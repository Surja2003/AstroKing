import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Animated, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { useRouter } from 'expo-router';
import ScreenWrapper from "../components/ScreenWrapper";
import { useUser } from "../context/user-context";
import { formatZodiacLabel, getZodiacFromDob } from '../lib/zodiac';
import { scheduleRetentionNotificationsAsync } from "../utils/notifications";
import { useTranslation } from 'react-i18next';
import { formatDobIsoToDisplay, normalizeDobDisplayInput, parseDobDisplayToIso } from '../lib/dob';
import { useAppColors } from '../lib/ui-theme';

export default function Login() {
  const router = useRouter();
  const { t } = useTranslation();
  const { user, setUser, language, setLanguage } = useUser();
  const c = useAppColors();
  const [name, setName] = useState("");
  const [dobDisplay, setDobDisplay] = useState("");

  useEffect(() => {
    // Pre-fill if we already have a profile.
    const existingName = String(user?.name ?? '').trim();
    const existingDob = String(user?.dob ?? '').trim();
    if (existingName) setName(existingName);
    if (existingDob) setDobDisplay(formatDobIsoToDisplay(existingDob) || '');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const dobParsed = useMemo(() => parseDobDisplayToIso(dobDisplay), [dobDisplay]);
  const dobIso = dobParsed?.iso ?? '';

  const zodiac = useMemo(() => getZodiacFromDob(dobIso), [dobIso]);
  const zodiacLabel = useMemo(() => formatZodiacLabel(zodiac), [zodiac]);

  const zodiacAnim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.spring(zodiacAnim, {
      toValue: zodiac ? 1 : 0,
      useNativeDriver: true,
      damping: 16,
      stiffness: 180,
      mass: 0.8,
    }).start();
  }, [zodiac, zodiacAnim]);

  const canContinue = useMemo(() => {
    const hasName = !!name.trim();
    const hasDob = !!dobParsed?.iso;
    return hasName && hasDob;
  }, [dobParsed?.iso, name]);

  const onDobChange = (text: string) => {
    setDobDisplay(normalizeDobDisplayInput(text));
  };

  return (
    <ScreenWrapper>
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={[styles.title, { color: c.text }]}>{t('login.title')}</Text>
          <Text style={[styles.subtitle, { color: c.muted }]}>
            {t('login.subtitle')}
          </Text>
        </View>

        <View style={[styles.form, { backgroundColor: c.card, borderColor: c.cardBorder }]}>
          <Text style={[styles.label, { color: c.textSecondary }]}>{t('login.nameLabel')}</Text>
          <TextInput
            placeholder={t('login.namePlaceholder')}
            placeholderTextColor={c.placeholder}
            style={[styles.input, { borderColor: c.inputBorder, backgroundColor: c.inputBg, color: c.text }]}
            value={name}
            onChangeText={setName}
            autoCapitalize="words"
            returnKeyType="next"
          />

          <Text style={[styles.label, { color: c.textSecondary }]}>{t('login.dobLabel')}</Text>
          <TextInput
            placeholder={t('login.dobPlaceholder')}
            placeholderTextColor={c.placeholder}
            style={[styles.input, { borderColor: c.inputBorder, backgroundColor: c.inputBg, color: c.text }]}
            value={dobDisplay}
            onChangeText={onDobChange}
            keyboardType="numbers-and-punctuation"
            maxLength={10}
          />

          <Text style={[styles.label, { color: c.textSecondary }]}>{t('login.languageLabel')}</Text>
          <View style={styles.langRow}>
            {([
              { key: 'en', label: t('login.languageEnglish') },
              { key: 'hi', label: t('login.languageHindi') },
              { key: 'bn', label: t('login.languageBengali') },
            ] as any[]).map((opt) => {
              const active = language === opt.key;
              return (
                <Pressable
                  key={opt.key}
                  onPress={() => setLanguage(opt.key)}
                  style={({ pressed }) => [
                    styles.langChip,
                    {
                      borderColor: active ? c.badgeBorder : c.inputBorder,
                      backgroundColor: active ? c.primarySoft : c.bg2,
                    },
                    pressed ? { opacity: 0.92 } : null,
                  ]}
                >
                  <Text
                    style={[
                      styles.langText,
                      { color: active ? c.primary : c.textSecondary },
                    ]}
                  >
                    {opt.label}
                  </Text>
                </Pressable>
              );
            })}
          </View>

          <Animated.View
            style={[
              styles.zodiacPill,
              {
                backgroundColor: c.badgeBg,
                borderColor: c.badgeBorder,
              },
              {
                opacity: zodiacAnim,
                transform: [
                  {
                    scale: zodiacAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: [0.96, 1],
                    }),
                  },
                ],
              },
            ]}
          >
            <Text style={[styles.zodiacText, { color: c.badgeText }]}>
              {zodiacLabel ? `${zodiacLabel} detected` : ''}
            </Text>
          </Animated.View>

          <Pressable
            style={({ pressed }) => [
              styles.continue,
              { backgroundColor: c.primary, shadowColor: c.primary },
              !canContinue && styles.continueDisabled,
              pressed && canContinue ? { opacity: 0.9 } : null,
            ]}
            disabled={!canContinue}
            onPress={() => {
              const trimmedName = name.trim();
              const trimmedDob = dobIso;
              setUser({ name: trimmedName, dob: trimmedDob });
              void scheduleRetentionNotificationsAsync();
              // Land on the real “Home” tab.
              router.replace({ pathname: '/(tabs)' });
            }}
          >
            <Text style={styles.continueText}>{t('login.continue')}</Text>
          </Pressable>

          <Text style={[styles.helper, { color: c.muted }]}>
            {t('login.tip')}
          </Text>
        </View>
      </View>
    </ScreenWrapper>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center' },
  header: {
    marginBottom: 24,
    gap: 10,
  },
  input: {
    borderWidth: 1,
    borderRadius: 12,
    marginBottom: 15,
    padding: 12,
  },
  title: {
    fontSize: 28,
    textAlign: 'center',
    fontWeight: '800',
    letterSpacing: -0.2,
  },
  subtitle: {
    textAlign: 'center',
    lineHeight: 20,
  },
  form: {
    borderWidth: 1,
    borderRadius: 18,
    padding: 16,
  },
  label: {
    fontWeight: '700',
    marginBottom: 8,
  },
  zodiacPill: {
    alignSelf: 'center',
    borderWidth: 1,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
    marginTop: 4,
    marginBottom: 14,
  },
  zodiacText: {
    fontWeight: '800',
  },
  continue: {
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: 'center',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.7,
    shadowRadius: 12,
    elevation: 10,
  },
  continueDisabled: {
    opacity: 0.45,
  },
  continueText: {
    color: 'white',
    fontWeight: '800',
    fontSize: 16,
  },
  helper: {
    marginTop: 12,
    textAlign: 'center',
  },
  langRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 16,
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  langChip: {
    borderWidth: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 999,
  },
  langText: {
    fontWeight: '800',
  },
});
