import axios from 'axios';
import Constants from 'expo-constants';
import * as Device from 'expo-device';
import { Platform } from 'react-native';

function getExpoHost() {
  const hostUri =
    Constants.expoConfig?.hostUri ??
    Constants.manifest2?.extra?.expoClient?.hostUri ??
    Constants.manifest?.hostUri;

  if (!hostUri || typeof hostUri !== 'string') return null;

  // hostUri usually looks like: "10.120.106.141:8081"
  const withoutScheme = hostUri.replace(/^https?:\/\//, '');
  const host = withoutScheme.split(':')[0];
  return host || null;
}

function isIpAddress(host) {
  return /^\d{1,3}(?:\.\d{1,3}){3}$/.test(host);
}

export function getApiBaseUrl() {
  const env = process.env.EXPO_PUBLIC_API_URL;
  if (env) return env;

  // Web should hit localhost.
  if (Platform.OS === 'web') return 'http://localhost:8000';

  // Android emulator should hit the host machine via 10.0.2.2.
  // This avoids flaky LAN IP routing inside the emulator.
  if (Platform.OS === 'android' && !Device.isDevice) {
    return 'http://10.0.2.2:8000';
  }

  // Physical device on LAN: use the Expo dev server host IP.
  const host = getExpoHost();
  if (host) {
    // iOS simulator (on macOS) can use localhost; for other cases we still try.
    if (host === 'localhost' || host === '127.0.0.1') {
      return 'http://localhost:8000';
    }

    // If Expo is running in tunnel mode, host may be a domain; backend usually isn't tunneled.
    if (!isIpAddress(host)) {
      // eslint-disable-next-line no-console
      console.log(
        `[API] Expo host is "${host}" (likely tunnel). Set EXPO_PUBLIC_API_URL=http://<YOUR_LAN_IP>:8000 for device testing.`
      );
    }

    return `http://${host}:8000`;
  }

  // Fallback (works for iOS simulator, not for physical device).
  return 'http://127.0.0.1:8000';
}

export const api = axios.create({
  baseURL: getApiBaseUrl(),
  timeout: 15000,
});
