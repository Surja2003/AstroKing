import { getItem, setItem } from './storage';

export type LocalPalmHistoryEntry = {
  id: string;
  created_at: string;
  local_uri: string;
  message: string;
};

const storageKeyForUser = (name: string, dob: string) => {
  const n = (name || '').trim();
  const d = (dob || '').trim();
  return `palm_history_v1:${n}:${d}`;
};

const safeJsonParse = <T>(raw: string | null): T | null => {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
};

const tryPersistImage = async (uri: string, id: string): Promise<string> => {
  try {
    const FileSystemImport = await import('expo-file-system');
    const FileSystem: any = (FileSystemImport as any)?.default ?? FileSystemImport;
    const base = FileSystem?.documentDirectory;
    if (!base) return uri;

    const dir = `${base}palm_history`;
    try {
      await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
    } catch {
      // ignore
    }

    // Try to infer extension; default to jpg.
    const m = /\.([a-zA-Z0-9]+)(\?.*)?$/.exec(uri);
    const ext = (m?.[1] || 'jpg').toLowerCase();
    const target = `${dir}/palm_${id}.${ext}`;

    await FileSystem.copyAsync({ from: uri, to: target });
    return target;
  } catch {
    return uri;
  }
};

export const loadLocalPalmHistory = async (name: string, dob: string, limit = 20): Promise<LocalPalmHistoryEntry[]> => {
  const key = storageKeyForUser(name, dob);
  const raw = await getItem(key);
  const parsed = safeJsonParse<LocalPalmHistoryEntry[]>(raw) || [];
  const items = Array.isArray(parsed) ? parsed : [];

  items.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
  return items.slice(0, Math.max(1, limit));
};

export const appendLocalPalmHistory = async (name: string, dob: string, params: { imageUri: string; message: string }) => {
  const key = storageKeyForUser(name, dob);

  const created_at = new Date().toISOString();
  const id = String(Date.now());
  const local_uri = await tryPersistImage(params.imageUri, id);

  const entry: LocalPalmHistoryEntry = {
    id,
    created_at,
    local_uri,
    message: String(params.message || '').trim(),
  };

  const raw = await getItem(key);
  const parsed = safeJsonParse<LocalPalmHistoryEntry[]>(raw) || [];
  const items = Array.isArray(parsed) ? parsed : [];

  const next = [entry, ...items].slice(0, 25);
  await setItem(key, JSON.stringify(next));

  return entry;
};
