import React, { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { Animated, StyleSheet } from 'react-native';
import type { StyleProp, ViewStyle } from 'react-native';

type FadeInProps = {
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
  durationMs?: number;
  from?: number;
  to?: number;
};

export default function FadeIn({ children, style, durationMs = 300, from = 0, to = 1 }: FadeInProps) {
  const opacity = useRef(new Animated.Value(from)).current;

  useEffect(() => {
    opacity.setValue(from);
    Animated.timing(opacity, {
      toValue: to,
      duration: durationMs,
      useNativeDriver: true,
    }).start();
  }, [durationMs, from, opacity, to]);

  return <Animated.View style={[styles.base, style, { opacity }]}>{children}</Animated.View>;
}

const styles = StyleSheet.create({
  base: {
    flexShrink: 1,
  },
});
