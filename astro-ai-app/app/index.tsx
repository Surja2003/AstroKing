import React from 'react';
import { Redirect } from 'expo-router';
import { useUser } from '@/context/user-context';

export default function Index() {
  const { hydrated, user } = useUser();

  if (!hydrated) return null;

  const hasProfile = !!String(user?.name ?? '').trim() && !!String(user?.dob ?? '').trim();
  return <Redirect href={hasProfile ? '/(tabs)' : '/login'} />;
}
