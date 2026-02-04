import React, { useMemo, useState } from 'react';
import {
  View,
  Text,
  TextInput,
  Pressable,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useUser } from '../context/user-context';
import { useTranslation } from 'react-i18next';
import { formatDobIsoToDisplay, normalizeDobDisplayInput, parseDobDisplayToIso } from '../lib/dob';
import { useAppColors } from '../lib/ui-theme';

export default function LoginScreen({ navigation } = {}) {
  const router = useRouter();
  const { t } = useTranslation();
  const c = useAppColors();

  const { user, setUser, language, setLanguage } = useUser();
  const [name, setName] = useState(user?.name ?? '');
  const [dobDisplay, setDobDisplay] = useState(formatDobIsoToDisplay(user?.dob ?? '') || '');

  const nav = useMemo(() => {
    if (navigation) return navigation;
    const routeMap = {
      Home: '/(tabs)',
      Chat: '/(tabs)/explore',
      Login: '/login',
    };
    return {
      navigate: (name) => router.push(routeMap[name] ?? '/(tabs)'),
      replace: (name) => router.replace(routeMap[name] ?? '/(tabs)'),
    };
  }, [navigation, router]);

  const onLogin = () => {
    const parsed = parseDobDisplayToIso(dobDisplay.trim());
    if (!parsed?.iso) return;
    setUser({ name: name.trim(), dob: parsed.iso });
    nav.replace?.('Home');
  };

  return (
    <SafeAreaView style={[styles.safe, { backgroundColor: c.bg }]}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.safe}
      >
        <View style={[styles.container, { backgroundColor: c.bg }]}>
          <Text style={[styles.title, { color: c.text }]}>{t('login.title')}</Text>
          <Text style={[styles.subtitle, { color: c.muted }]}>{t('login.subtitle')}</Text>

          <View style={styles.field}>
            <Text style={[styles.label, { color: c.textSecondary }]}>{t('login.nameLabel')}</Text>
            <TextInput
              value={name}
              onChangeText={setName}
              placeholder={t('login.namePlaceholder')}
              placeholderTextColor={c.placeholder}
              autoCapitalize="words"
              style={[styles.input, { borderColor: c.inputBorder, backgroundColor: c.inputBg, color: c.text }]}
            />
          </View>

          <View style={styles.field}>
            <Text style={[styles.label, { color: c.textSecondary }]}>{t('login.dobLabel')}</Text>
            <TextInput
              value={dobDisplay}
              onChangeText={(txt) => setDobDisplay(normalizeDobDisplayInput(txt))}
              placeholder={t('login.dobPlaceholder')}
              placeholderTextColor={c.placeholder}
              autoCapitalize="none"
              keyboardType={Platform.OS === 'ios' ? 'numbers-and-punctuation' : 'default'}
              maxLength={10}
              style={[styles.input, { borderColor: c.inputBorder, backgroundColor: c.inputBg, color: c.text }]}
            />
          </View>

          <View style={styles.field}>
            <Text style={[styles.label, { color: c.textSecondary }]}>{t('login.languageLabel')}</Text>
            <View style={{ flexDirection: 'row', gap: 10, flexWrap: 'wrap' }}>
              {[
                { key: 'en', label: t('login.languageEnglish') },
                { key: 'hi', label: t('login.languageHindi') },
                { key: 'bn', label: t('login.languageBengali') },
              ].map((opt) => {
                const active = language === opt.key;
                return (
                  <Pressable
                    key={opt.key}
                    onPress={() => setLanguage(opt.key)}
                    style={[
                      {
                        borderWidth: 1,
                        borderColor: active ? c.accentBorderStrong : c.inputBorder,
                        backgroundColor: active ? c.primarySoft : c.bg2,
                        paddingVertical: 8,
                        paddingHorizontal: 10,
                        borderRadius: 999,
                      },
                    ]}
                  >
                    <Text style={{ color: active ? c.primary : c.textSecondary, fontWeight: '700' }}>{opt.label}</Text>
                  </Pressable>
                );
              })}
            </View>
          </View>

          <Pressable onPress={onLogin} style={[styles.button, { backgroundColor: c.primary }]}>
            <Text style={[styles.buttonText, { color: c.onPrimary }]}>{t('login.continue')}</Text>
          </Pressable>

          <Pressable onPress={() => nav.navigate?.('Home')} style={styles.link}>
            <Text style={[styles.linkText, { color: c.primary }]}>Continue to Home</Text>
          </Pressable>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1 },
  container: { flex: 1, padding: 20, justifyContent: 'center', gap: 12 },
  title: {
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 8,
  },
  subtitle: {
    marginTop: -8,
    marginBottom: 10,
  },
  field: {
    gap: 6,
  },
  label: {
    fontSize: 13,
    fontWeight: '600',
  },
  input: {
    borderWidth: 1,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
  },
  button: {
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 4,
  },
  buttonText: {
    fontWeight: '700',
    fontSize: 16,
  },
  link: {
    alignItems: 'center',
    paddingVertical: 8,
  },
  linkText: {
    fontWeight: '600',
  },
});
