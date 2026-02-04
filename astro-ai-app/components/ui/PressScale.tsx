import React, { useMemo, useRef } from 'react';
import type { ReactNode } from 'react';
import { Animated, Pressable, StyleSheet } from 'react-native';
import type { StyleProp, ViewStyle } from 'react-native';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

type PressScaleProps = {
  children: ReactNode;
  onPress?: () => void;
  disabled?: boolean;
  style?: StyleProp<ViewStyle>;
  hitSlop?: { top?: number; left?: number; right?: number; bottom?: number };
  accessibilityLabel?: string;
  androidRippleColor?: string;
};

export default function PressScale({
  children,
  onPress,
  disabled,
  style,
  hitSlop,
  accessibilityLabel,
  androidRippleColor,
}: PressScaleProps) {
  const scale = useRef(new Animated.Value(1)).current;

  const springConfig = useMemo(
    () => ({
      useNativeDriver: true,
      speed: 22,
      bounciness: 5,
    }),
    []
  );

  const animateTo = (toValue: number) => {
    Animated.spring(scale, { toValue, ...springConfig }).start();
  };

  return (
    <AnimatedPressable
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel}
      hitSlop={hitSlop}
      disabled={disabled}
      onPress={onPress}
      android_ripple={androidRippleColor ? { color: androidRippleColor, borderless: false } : undefined}
      onPressIn={() => {
        if (disabled) return;
        animateTo(0.96);
      }}
      onPressOut={() => {
        animateTo(1);
      }}
      style={[styles.base, style, { transform: [{ scale }] }]}
    >
      {children}
    </AnimatedPressable>
  );
}

const styles = StyleSheet.create({
  base: {
    overflow: 'hidden',
  },
});
