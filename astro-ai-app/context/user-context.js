import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import i18n from '../lib/i18n';
import { getItem, setItem } from '../lib/storage';

const STORAGE_KEY = 'astroking:user_v1';

const DEFAULT_USER = { name: '', dob: '' };
const DEFAULT_LANGUAGE = 'en';

const UserContext = createContext({
  hydrated: false,
  user: { name: '', dob: '' },
  language: 'en',
  setUser: (_next) => {},
  setLanguage: (_lang) => {},
  clearUser: () => {},
});

export function UserProvider({ children }) {
  const [hydrated, setHydrated] = useState(false);
  const [user, setUserState] = useState(DEFAULT_USER);
  const [language, setLanguageState] = useState(DEFAULT_LANGUAGE);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const raw = await getItem(STORAGE_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        const nextUser = parsed?.user ?? parsed ?? null;
        const nextLang = parsed?.language ?? DEFAULT_LANGUAGE;

        if (!cancelled) {
          if (nextUser && typeof nextUser === 'object') {
            setUserState({
              name: String(nextUser?.name ?? ''),
              dob: String(nextUser?.dob ?? ''),
            });
          }
          if (typeof nextLang === 'string' && nextLang) {
            setLanguageState(nextLang);
            try {
              void i18n.changeLanguage(nextLang);
            } catch {
              // ignore
            }
          }
        }
      } catch {
        // ignore
      } finally {
        if (!cancelled) setHydrated(true);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const persist = async (nextUser, nextLang) => {
    try {
      await setItem(
        STORAGE_KEY,
        JSON.stringify({
          user: nextUser ?? DEFAULT_USER,
          language: nextLang ?? DEFAULT_LANGUAGE,
        })
      );
    } catch {
      // ignore
    }
  };

  const value = useMemo(
    () => ({
      hydrated,
      user,
      language,
      setUser: (next) => {
        setUserState((prev) => {
          const merged = { ...prev, ...(next ?? {}) };
          void persist(merged, language);
          return merged;
        });
      },
      setLanguage: (lang) => {
        const nextLang = String(lang ?? '').trim() || DEFAULT_LANGUAGE;
        setLanguageState(nextLang);
        try {
          void i18n.changeLanguage(nextLang);
        } catch {
          // ignore
        }
        void persist(user, nextLang);
      },
      clearUser: () => {
        setUserState(DEFAULT_USER);
        void persist(DEFAULT_USER, language);
      },
    }),
    [hydrated, language, user]
  );

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

export function useUser() {
  return useContext(UserContext);
}
