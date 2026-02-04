import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { Stack } from 'expo-router';
import { useRouter } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import 'react-native-reanimated';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';

import { useColorScheme } from '@/hooks/use-color-scheme';
import { UserProvider } from '@/context/user-context';
import '@/lib/i18n';
import { getAppColors } from '@/lib/ui-theme';

export const unstable_settings = {
  anchor: '(tabs)',
};

export default function RootLayout() {
  const colorScheme = useColorScheme();
  const router = useRouter();
  const { t } = useTranslation();
  const isDark = colorScheme === 'dark';
  const c = getAppColors(colorScheme);

  useEffect(() => {
    let sub: any;
    (async () => {
      const Notifications = await import('expo-notifications');
      sub = Notifications.addNotificationResponseReceivedListener((response: any) => {
        const route = response?.notification?.request?.content?.data?.route;
        if (route === '/(tabs)' || route === '/weekly' || route === '/history' || route === '/camera' || route === '/chat') {
          router.push(route);
        }
      });
    })();

    return () => {
      try {
        sub?.remove?.();
      } catch {
        // ignore
      }
    };
  }, [router]);

  return (
    <UserProvider>
      <ThemeProvider value={isDark ? DarkTheme : DefaultTheme}>
        <>
          <StatusBar style={isDark ? 'light' : 'dark'} />
          <Stack
            screenOptions={{
              headerStyle: { backgroundColor: isDark ? c.bg : c.bg2 },
              headerTintColor: c.text,
              headerTitleStyle: { fontWeight: 'bold' },
              headerShadowVisible: false,
            }}
          >
            <Stack.Screen name="login" options={{ headerShown: false }} />
            <Stack.Screen name="home" options={{ title: t('nav.dailyInsight') }} />
            <Stack.Screen name="chat" options={{ title: t('nav.aiAstroChat') }} />
            <Stack.Screen name="history" options={{ title: t('nav.yourHistory') }} />
            <Stack.Screen name="weekly" options={{ title: t('nav.weeklyReport') }} />
            <Stack.Screen
              name="camera"
              options={{
                title: t('nav.palmScan'),
                presentation: 'modal',
              }}
            />
            <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
            <Stack.Screen name="modal" options={{ presentation: 'modal', title: 'Modal' }} />
          </Stack>
        </>
      </ThemeProvider>
    </UserProvider>
  );
}
