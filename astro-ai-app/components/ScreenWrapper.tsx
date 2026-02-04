import React from 'react';
import type { StyleProp, ViewStyle } from 'react-native';
import { StyleSheet, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useColorScheme } from '../hooks/use-color-scheme';
import { useAppColors } from '../lib/ui-theme';

type ScreenWrapperProps = {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
};

export default function ScreenWrapper({ children, style }: ScreenWrapperProps) {
  const colorScheme = useColorScheme();
  const c = useAppColors();

  if (colorScheme !== 'dark') {
    return (
      <View style={[styles.container, { backgroundColor: c.bg }, style]}>
        {children}
      </View>
    );
  }

  return (
    <LinearGradient colors={["#0B0F1A", "#121826", "#0B0F1A"]} style={[styles.container, style]}>
      {children}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20 },
});
