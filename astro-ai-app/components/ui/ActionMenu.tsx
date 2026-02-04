import React, { useEffect, useMemo, useRef } from 'react';
import type { ReactNode } from 'react';
import { Animated, Easing, Modal, Pressable, StyleSheet, Text, View } from 'react-native';

import { useColorScheme } from '../../hooks/use-color-scheme';
import { getAppColors, shadowStyle } from '../../lib/ui-theme';

export type ActionMenuItem = {
  key: string;
  label: string;
  onPress: () => void;
  disabled?: boolean;
  destructive?: boolean;
  icon?: ReactNode;
};

type ActionMenuProps = {
  visible: boolean;
  onClose: () => void;
  title?: string;
  items: ActionMenuItem[];
};

export default function ActionMenu({ visible, onClose, title, items }: ActionMenuProps) {
  const colorScheme = useColorScheme();
  const c = getAppColors(colorScheme);
  const translateY = useRef(new Animated.Value(24)).current;
  const backdropOpacity = useRef(new Animated.Value(0)).current;

  const openAnim = useMemo(
    () =>
      Animated.parallel([
        Animated.timing(backdropOpacity, {
          toValue: 1,
          duration: 140,
          easing: Easing.out(Easing.quad),
          useNativeDriver: true,
        }),
        Animated.timing(translateY, {
          toValue: 0,
          duration: 250,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
      ]),
    [backdropOpacity, translateY]
  );

  const closeAnim = useMemo(
    () =>
      Animated.parallel([
        Animated.timing(backdropOpacity, {
          toValue: 0,
          duration: 120,
          easing: Easing.in(Easing.quad),
          useNativeDriver: true,
        }),
        Animated.timing(translateY, {
          toValue: 24,
          duration: 200,
          easing: Easing.in(Easing.cubic),
          useNativeDriver: true,
        }),
      ]),
    [backdropOpacity, translateY]
  );

  useEffect(() => {
    if (visible) {
      openAnim.start();
    } else {
      backdropOpacity.setValue(0);
      translateY.setValue(24);
    }
  }, [backdropOpacity, openAnim, translateY, visible]);

  const requestClose = () => {
    closeAnim.start(({ finished }) => {
      if (finished) onClose();
    });
  };

  return (
    <Modal visible={visible} transparent animationType="none" onRequestClose={requestClose}>
      <Animated.View style={[styles.backdrop, { opacity: backdropOpacity }]}>
        <Pressable style={StyleSheet.absoluteFillObject} onPress={requestClose} />
      </Animated.View>
      <View style={styles.sheetWrap} pointerEvents="box-none">
        <Animated.View
          style={[
            styles.sheet,
            { backgroundColor: c.bg2, borderColor: c.cardBorder },
            shadowStyle(colorScheme),
            { transform: [{ translateY }] },
          ]}
        >
          {!!title && <Text style={[styles.title, { color: c.text }]}>{title}</Text>}
          {items.map((item) => {
            const color = item.destructive ? c.danger : c.text;
            return (
              <Pressable
                key={item.key}
                disabled={item.disabled}
                onPress={() => {
                  if (item.disabled) return;
                  requestClose();
                  item.onPress();
                }}
                style={({ pressed }) => [
                  styles.item,
                  pressed && !item.disabled && { backgroundColor: c.surfaceSubtle },
                  item.disabled && { opacity: 0.5 },
                ]}
              >
                <View style={styles.itemRow}>
                  {!!item.icon && <View style={styles.icon}>{item.icon}</View>}
                  <Text style={[styles.itemLabel, { color }]}>{item.label}</Text>
                </View>
              </Pressable>
            );
          })}
        </Animated.View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  sheetWrap: {
    flex: 1,
    justifyContent: 'flex-end',
    padding: 14,
  },
  sheet: {
    borderWidth: 1,
    borderRadius: 16,
    paddingVertical: 8,
    overflow: 'hidden',
  },
  title: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    fontSize: 13,
    letterSpacing: 0.2,
    fontWeight: '800',
    opacity: 0.9,
  },
  item: {
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  itemRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  icon: {
    width: 18,
    height: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  itemLabel: {
    fontSize: 15,
    fontWeight: '700',
  },
});
