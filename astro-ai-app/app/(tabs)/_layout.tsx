import { Tabs } from 'expo-router';
import React from 'react';
import { Platform, View } from 'react-native';
import { useTranslation } from 'react-i18next';

import { HapticTab } from '@/components/haptic-tab';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { getAppColors } from '@/lib/ui-theme';

export default function TabLayout() {
  const colorScheme = useColorScheme();
  const { t } = useTranslation();
  const c = getAppColors(colorScheme);

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: c.primary,
        tabBarInactiveTintColor: c.muted,
        headerShown: false,
        tabBarButton: HapticTab,
        tabBarStyle: {
          backgroundColor: c.bg2,
          borderTopColor: c.border,
        },
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: t('nav.home'),
          tabBarIcon: ({ color, focused }) => {
            const glowStyle = focused
              ? Platform.select({
                  web: {
                    boxShadow: `0 0 16px ${c.primarySoft}`,
                  } as any,
                  default: {
                    shadowColor: c.primary,
                    shadowOpacity: 0.45,
                    shadowRadius: 10,
                    shadowOffset: { width: 0, height: 4 },
                    elevation: 6,
                  },
                })
              : null;

            return (
              <View
                style={[
                  {
                    padding: 6,
                    borderRadius: 999,
                    backgroundColor: focused ? c.primarySoft : 'transparent',
                  },
                  glowStyle,
                ]}
              >
                <IconSymbol size={28} name="house.fill" color={color} />
              </View>
            );
          },
        }}
      />

      <Tabs.Screen
        name="camera"
        options={{
          title: t('nav.palmScan'),
          tabBarIcon: ({ color, focused }) => {
            const glowStyle = focused
              ? Platform.select({
                  web: {
                    boxShadow: `0 0 16px ${c.primarySoft}`,
                  } as any,
                  default: {
                    shadowColor: c.primary,
                    shadowOpacity: 0.45,
                    shadowRadius: 10,
                    shadowOffset: { width: 0, height: 4 },
                    elevation: 6,
                  },
                })
              : null;

            return (
              <View
                style={[
                  {
                    padding: 6,
                    borderRadius: 999,
                    backgroundColor: focused ? c.primarySoft : 'transparent',
                  },
                  glowStyle,
                ]}
              >
                <IconSymbol size={28} name="camera.fill" color={color} />
              </View>
            );
          },
        }}
      />

      <Tabs.Screen
        name="explore"
        options={{
          title: t('nav.chat'),
          tabBarIcon: ({ color, focused }) => {
            const glowStyle = focused
              ? Platform.select({
                  web: {
                    boxShadow: `0 0 16px ${c.primarySoft}`,
                  } as any,
                  default: {
                    shadowColor: c.primary,
                    shadowOpacity: 0.45,
                    shadowRadius: 10,
                    shadowOffset: { width: 0, height: 4 },
                    elevation: 6,
                  },
                })
              : null;

            return (
              <View
                style={[
                  {
                    padding: 6,
                    borderRadius: 999,
                    backgroundColor: focused ? c.primarySoft : 'transparent',
                  },
                  glowStyle,
                ]}
              >
                <IconSymbol size={28} name="message.fill" color={color} />
              </View>
            );
          },
        }}
      />
    </Tabs>
  );
}
