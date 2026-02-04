import { NativeModules, Platform } from 'react-native';
import { requireOptionalNativeModule } from 'expo-modules-core';

let _asyncStoragePromise = null;
let _warnedAsyncStorageUnavailable = false;

let _secureStorePromise = null;
let _warnedSecureStoreUnavailable = false;

let _fileSystemPromise = null;
let _warnedFileSystemUnavailable = false;

const FILESTORE_NAME = 'astroking_kv_v1.json';
let _fileStoreCache = null;
let _fileStoreLoadPromise = null;
let _fileStoreWritePromise = Promise.resolve();

function warnAsyncStorageUnavailableOnce(e) {
  if (_warnedAsyncStorageUnavailable) return;
  _warnedAsyncStorageUnavailable = true;

  const msg = e?.message ? String(e.message) : String(e ?? 'Unknown error');
  // eslint-disable-next-line no-console
  console.warn(
    [
      'AsyncStorage native module is unavailable (likely Expo Go).',
      'Falling back to a non-crashing storage implementation (SecureStore/memory).',
      'To enable real AsyncStorage persistence: build and run a dev client:',
      '  - npx expo run:android (or run:ios)',
      '  - npx expo start --dev-client',
      `Original error: ${msg}`,
    ].join('\n')
  );
}

function warnSecureStoreUnavailableOnce(e) {
  if (_warnedSecureStoreUnavailable) return;
  _warnedSecureStoreUnavailable = true;

  const msg = e?.message ? String(e.message) : String(e ?? 'Unknown error');
  // eslint-disable-next-line no-console
  console.warn(
    [
      'SecureStore native module is unavailable (ExpoSecureStore).',
      'Falling back to in-memory storage (non-persistent).',
      'If you are using a dev client, rebuild it after installing Expo modules:',
      '  - npx expo run:android (or run:ios)',
      '  - npx expo start --dev-client',
      `Original error: ${msg}`,
    ].join('\n')
  );
}

function warnFileSystemUnavailableOnce(e) {
  if (_warnedFileSystemUnavailable) return;
  _warnedFileSystemUnavailable = true;

  const msg = e?.message ? String(e.message) : String(e ?? 'Unknown error');
  // eslint-disable-next-line no-console
  console.warn(
    [
      'FileSystem storage fallback is unavailable (ExponentFileSystem).',
      'Falling back to in-memory storage (non-persistent).',
      `Original error: ${msg}`,
    ].join('\n')
  );
}

async function getAsyncStorage() {
  if (_asyncStoragePromise) return _asyncStoragePromise;

  // If the native module isn't present in this binary, don't even try to import
  // the JS module (it will throw during evaluation and surface as a red screen).
  // This can happen when running an older dev client / Expo Go mismatch.
  const hasNative = !!NativeModules?.RNCAsyncStorage;
  if (!hasNative) {
    _asyncStoragePromise = Promise.resolve(null);
    return _asyncStoragePromise;
  }

  // Delay importing until runtime so we can provide a clearer error message
  // when someone runs the app in Expo Go (where native modules may be missing).
  _asyncStoragePromise = import('@react-native-async-storage/async-storage')
    .then((m) => m.default ?? m)
    .catch((e) => {
      warnAsyncStorageUnavailableOnce(e);
      return null;
    });
  return _asyncStoragePromise;
}

async function getSecureStore() {
  if (_secureStorePromise) return _secureStorePromise;

  // Avoid importing the package JS if the native module isn't present.
  const hasNative = !!requireOptionalNativeModule?.('ExpoSecureStore');
  if (!hasNative) {
    _secureStorePromise = Promise.resolve(null);
    return _secureStorePromise;
  }

  // Lazy import to avoid crashing the whole app when the native module isn't present.
  _secureStorePromise = import('expo-secure-store')
    .then((m) => m)
    .catch((e) => {
      warnSecureStoreUnavailableOnce(e);
      return null;
    });
  return _secureStorePromise;
}

async function getFileSystem() {
  if (_fileSystemPromise) return _fileSystemPromise;

  const hasNative = !!requireOptionalNativeModule?.('ExponentFileSystem');
  if (!hasNative) {
    _fileSystemPromise = Promise.resolve(null);
    return _fileSystemPromise;
  }

  _fileSystemPromise = import('expo-file-system')
    .then((m) => m)
    .catch((e) => {
      warnFileSystemUnavailableOnce(e);
      return null;
    });
  return _fileSystemPromise;
}

async function loadFileStore() {
  if (_fileStoreCache) return _fileStoreCache;
  if (_fileStoreLoadPromise) return _fileStoreLoadPromise;

  _fileStoreLoadPromise = (async () => {
    const FileSystem = await getFileSystem();
    if (!FileSystem?.documentDirectory) {
      _fileStoreCache = {};
      return _fileStoreCache;
    }

    const fileUri = `${FileSystem.documentDirectory}${FILESTORE_NAME}`;
    try {
      const info = await FileSystem.getInfoAsync(fileUri);
      if (!info?.exists) {
        _fileStoreCache = {};
        return _fileStoreCache;
      }

      const raw = await FileSystem.readAsStringAsync(fileUri);
      const parsed = raw ? JSON.parse(raw) : {};
      _fileStoreCache = parsed && typeof parsed === 'object' ? parsed : {};
      return _fileStoreCache;
    } catch (e) {
      warnFileSystemUnavailableOnce(e);
      _fileStoreCache = {};
      return _fileStoreCache;
    } finally {
      _fileStoreLoadPromise = null;
    }
  })();

  return _fileStoreLoadPromise;
}

async function writeFileStore(nextStore) {
  const FileSystem = await getFileSystem();
  if (!FileSystem?.documentDirectory) return;

  const fileUri = `${FileSystem.documentDirectory}${FILESTORE_NAME}`;
  const payload = JSON.stringify(nextStore ?? {});

  // Serialize writes to avoid clobbering.
  _fileStoreWritePromise = _fileStoreWritePromise
    .catch(() => {})
    .then(() => FileSystem.writeAsStringAsync(fileUri, payload));
  await _fileStoreWritePromise;
}

// In-memory fallback (e.g. when running in environments without SecureStore).
const mem = new Map();

/**
 * @param {string} k
 * @returns {Promise<string|null>}
 */
export async function getItem(k) {
  if (!k) return null;

  if (Platform.OS === 'web') {
    try {
      return globalThis?.localStorage?.getItem(k) ?? null;
    } catch {
      return mem.get(k) ?? null;
    }
  }

  try {
    const AsyncStorage = await getAsyncStorage();
    if (AsyncStorage?.getItem) {
      return (await AsyncStorage.getItem(k)) ?? null;
    }

    try {
      const SecureStore = await getSecureStore();
      if (SecureStore?.getItemAsync) {
        return (await SecureStore.getItemAsync(k)) ?? null;
      }
      const fsStore = await loadFileStore();
      return fsStore?.[k] ?? mem.get(k) ?? null;
    } catch {
      const fsStore = await loadFileStore();
      return fsStore?.[k] ?? mem.get(k) ?? null;
    }
  } catch (e) {
    warnAsyncStorageUnavailableOnce(e);
    try {
      const SecureStore = await getSecureStore();
      if (SecureStore?.getItemAsync) {
        return (await SecureStore.getItemAsync(k)) ?? null;
      }
      const fsStore = await loadFileStore();
      return fsStore?.[k] ?? mem.get(k) ?? null;
    } catch {
      const fsStore = await loadFileStore();
      return fsStore?.[k] ?? mem.get(k) ?? null;
    }
  }
}

/**
 * @param {string} k
 * @param {string} v
 */
export async function setItem(k, v) {
  if (!k) return;

  const next = typeof v === 'string' ? v : String(v ?? '');

  if (Platform.OS === 'web') {
    try {
      globalThis?.localStorage?.setItem(k, next);
      return;
    } catch {
      mem.set(k, next);
      return;
    }
  }

  try {
    const AsyncStorage = await getAsyncStorage();
    if (AsyncStorage?.setItem) {
      await AsyncStorage.setItem(k, next);
      return;
    }

    try {
      const SecureStore = await getSecureStore();
      if (SecureStore?.setItemAsync) {
        await SecureStore.setItemAsync(k, next);
      } else {
        const fsStore = await loadFileStore();
        fsStore[k] = next;
        _fileStoreCache = fsStore;
        try {
          await writeFileStore(fsStore);
        } catch {
          // ignore
        }
        mem.set(k, next);
      }
    } catch {
      const fsStore = await loadFileStore();
      fsStore[k] = next;
      _fileStoreCache = fsStore;
      try {
        await writeFileStore(fsStore);
      } catch {
        // ignore
      }
      mem.set(k, next);
    }
  } catch (e) {
    warnAsyncStorageUnavailableOnce(e);
    try {
      const SecureStore = await getSecureStore();
      if (SecureStore?.setItemAsync) {
        await SecureStore.setItemAsync(k, next);
      } else {
        const fsStore = await loadFileStore();
        fsStore[k] = next;
        _fileStoreCache = fsStore;
        try {
          await writeFileStore(fsStore);
        } catch {
          // ignore
        }
        mem.set(k, next);
      }
    } catch {
      const fsStore = await loadFileStore();
      fsStore[k] = next;
      _fileStoreCache = fsStore;
      try {
        await writeFileStore(fsStore);
      } catch {
        // ignore
      }
      mem.set(k, next);
    }
  }
}

/**
 * @param {string} k
 */
export async function removeItem(k) {
  if (!k) return;

  if (Platform.OS === 'web') {
    try {
      globalThis?.localStorage?.removeItem(k);
    } catch {
      // ignore
    }
    mem.delete(k);
    return;
  }

  try {
    const AsyncStorage = await getAsyncStorage();
    if (AsyncStorage?.removeItem) {
      await AsyncStorage.removeItem(k);
      return;
    }

    try {
      const SecureStore = await getSecureStore();
      if (SecureStore?.deleteItemAsync) {
        await SecureStore.deleteItemAsync(k);
      }
    } catch {
      // ignore
    }

    try {
      const fsStore = await loadFileStore();
      if (fsStore && typeof fsStore === 'object') {
        delete fsStore[k];
        _fileStoreCache = fsStore;
        await writeFileStore(fsStore);
      }
    } catch {
      // ignore
    }
    mem.delete(k);
  } catch (e) {
    warnAsyncStorageUnavailableOnce(e);
    try {
      const SecureStore = await getSecureStore();
      if (SecureStore?.deleteItemAsync) {
        await SecureStore.deleteItemAsync(k);
      }
    } catch {
      // ignore
    }

    try {
      const fsStore = await loadFileStore();
      if (fsStore && typeof fsStore === 'object') {
        delete fsStore[k];
        _fileStoreCache = fsStore;
        await writeFileStore(fsStore);
      }
    } catch {
      // ignore
    }
    mem.delete(k);
  }
}
