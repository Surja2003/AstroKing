import axios from 'axios';
import Constants from 'expo-constants';
import * as Device from 'expo-device';
import { Platform } from 'react-native';

const DEFAULT_API_URL = 'https://astroking.onrender.com';

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

  // Production default (global): use the hosted backend.
  return DEFAULT_API_URL;
}

export const api = axios.create({
  baseURL: getApiBaseUrl(),
  // Render free tier may take ~30-40s on the first request after sleeping.
  timeout: 60000,
});
