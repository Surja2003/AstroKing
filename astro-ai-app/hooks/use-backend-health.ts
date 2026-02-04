import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../lib/api';

type BackendHealthState = {
  serverOk: boolean;
  checking: boolean;
  lastCheckedAt?: number;
};

export function useBackendHealth(): BackendHealthState & { checkNow: () => void } {
  const [serverOk, setServerOk] = useState(true);
  const [checking, setChecking] = useState(false);
  const [lastCheckedAt, setLastCheckedAt] = useState<number | undefined>(undefined);

  const timerRef = useRef<any>(null);
  const cancelledRef = useRef(false);

  const check = useCallback(async () => {
    setChecking(true);
    try {
      await api.get('/health', { timeout: 2500 });
      if (!cancelledRef.current) setServerOk(true);
    } catch {
      if (!cancelledRef.current) setServerOk(false);
    } finally {
      if (!cancelledRef.current) {
        setLastCheckedAt(Date.now());
        setChecking(false);
      }
    }
  }, []);

  const scheduleNext = useCallback(
    (delayMs: number) => {
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(async () => {
        await check();
        // Fast retry when offline; slow heartbeat when online.
        scheduleNext(serverOk ? 30000 : 4000);
      }, delayMs);
    },
    [check, serverOk]
  );

  useEffect(() => {
    cancelledRef.current = false;

    (async () => {
      await check();
      scheduleNext(serverOk ? 30000 : 4000);
    })();

    return () => {
      cancelledRef.current = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const checkNow = useCallback(() => {
    void check();
  }, [check]);

  return { serverOk, checking, lastCheckedAt, checkNow };
}
