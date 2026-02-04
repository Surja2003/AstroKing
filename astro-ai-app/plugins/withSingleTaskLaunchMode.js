const { AndroidConfig, withAndroidManifest } = require('@expo/config-plugins');

/**
 * Ensures the main Activity uses launchMode="singleTask".
 * This prevents Android from launching multiple instances of the app when opening deep links,
 * which can lead to duplicate linking handlers (React Navigation / Expo Router warning).
 */
module.exports = function withSingleTaskLaunchMode(config) {
  return withAndroidManifest(config, (config) => {
    const manifest = config.modResults;

    const mainActivity = AndroidConfig.Manifest.getMainActivityOrThrow(manifest);
    mainActivity.$ = mainActivity.$ || {};
    mainActivity.$['android:launchMode'] = 'singleTask';

    return config;
  });
};
