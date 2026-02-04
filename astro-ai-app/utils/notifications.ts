import * as Device from 'expo-device';
import Constants from 'expo-constants';
import { Platform } from 'react-native';

type NotificationRoute = '/(tabs)' | '/weekly';

let handlerInitialized = false;

async function getNotifications() {
  return import('expo-notifications');
}

function getExpoProjectId(): string | undefined {
  // SDK 49+ may require projectId in some environments.
  const projectId =
    (Constants as any)?.easConfig?.projectId ??
    (Constants as any)?.expoConfig?.extra?.eas?.projectId;
  if (typeof projectId !== 'string') return undefined;
  const trimmed = projectId.trim();
  return trimmed.length ? trimmed : undefined;
}

export async function registerForPushNotificationsAsync(): Promise<string | undefined> {
  // Expo Go no longer supports remote push notifications for expo-notifications (SDK 53+).
  // Use a Development Build (dev client) for real push tokens.
  if ((Constants as any)?.appOwnership === 'expo') return undefined;

  // Push tokens require a physical device (not simulator).
  if (!Device.isDevice) return undefined;

  const Notifications = await getNotifications();

  ensureNotificationHandler(Notifications);

  // Android needs a channel to properly display notifications.
  if (Platform.OS === 'android') {
    await Notifications.setNotificationChannelAsync('default', {
      name: 'default',
      importance: Notifications.AndroidImportance.MAX,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: '#8ab4ff',
    });
  }

  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  let finalStatus = existingStatus;

  if (existingStatus !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }

  if (finalStatus !== 'granted') return undefined;

  const projectId = getExpoProjectId();
  if (!projectId) return undefined;

  try {
    const token = (await Notifications.getExpoPushTokenAsync({ projectId })).data;
    return token;
  } catch {
    return undefined;
  }
}

function ensureNotificationHandler(Notifications: any) {
  if (handlerInitialized) return;
  handlerInitialized = true;

  Notifications.setNotificationHandler({
    handleNotification: async () => ({
      shouldShowAlert: true,
      shouldPlaySound: false,
      shouldSetBadge: false,
      shouldShowBanner: true,
      shouldShowList: true,
    }),
  });
}

export async function ensureLocalNotificationPermissionsAsync(): Promise<boolean> {
  const Notifications = await getNotifications();
  ensureNotificationHandler(Notifications);

  // Android needs a channel to properly display notifications.
  if (Platform.OS === 'android') {
    await Notifications.setNotificationChannelAsync('default', {
      name: 'default',
      importance: Notifications.AndroidImportance.MAX,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: '#8ab4ff',
    });
  }

  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  let finalStatus = existingStatus;
  if (existingStatus !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }

  return finalStatus === 'granted';
}

export async function cancelRetentionNotificationsAsync(): Promise<void> {
  const Notifications = await getNotifications();
  // Simple approach: clear all scheduled notifications for this app.
  await Notifications.cancelAllScheduledNotificationsAsync();
}

async function scheduleNotification(
  Notifications: any,
  opts: {
    title: string;
    body: string;
    route: NotificationRoute;
    trigger: any;
  }
) {
  await Notifications.scheduleNotificationAsync({
    content: {
      title: opts.title,
      body: opts.body,
      sound: null,
      data: { route: opts.route },
    },
    trigger: opts.trigger,
  });
}

export async function scheduleRetentionNotificationsAsync(): Promise<void> {
  if (Platform.OS === 'web') return;

  const Notifications = await getNotifications();
  ensureNotificationHandler(Notifications);

  const ok = await ensureLocalNotificationPermissionsAsync();
  if (!ok) return;

  // Keep it deterministic: clear old schedules then re-add.
  await cancelRetentionNotificationsAsync();

  const dailyTrigger = (hour: number, minute: number) => ({ type: 'daily', hour, minute });
  const weeklyTrigger = (weekday: number, hour: number, minute: number) => ({ type: 'weekly', weekday, hour, minute });

  // Daily habit builders
  await scheduleNotification(Notifications, {
    title: 'Today’s focus is waiting ✨',
    body: 'Open your daily guidance and take one clean next step.',
    route: '/(tabs)',
    trigger: dailyTrigger(9, 0),
  });

  await scheduleNotification(Notifications, {
    title: 'Energy check-in',
    body: 'Your momentum shifts later today—review your focus card.',
    route: '/(tabs)',
    trigger: dailyTrigger(16, 0),
  });

  // Weekly report (Sunday)
  await scheduleNotification(Notifications, {
    title: 'Your Week in Review 📈',
    body: 'See your patterns, streak, and most-asked topic.',
    route: '/weekly',
    trigger: weeklyTrigger(1, 10, 0), // 1 = Sunday in expo-notifications
  });
}
