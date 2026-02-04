// Metro configuration for Expo
//
// Fixes Android bundling issues with certain ESM "module" builds in node_modules
// by preferring the CommonJS entry ("main") when available.

const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Prefer CJS builds over ESM "module" builds for compatibility.
// (Expo defaults can change across SDK versions; we pin this for stability.)
config.resolver = config.resolver || {};
config.resolver.mainFields = ['react-native', 'browser', 'main'];

module.exports = config;
