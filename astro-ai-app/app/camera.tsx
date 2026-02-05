import { CameraView, useCameraPermissions } from 'expo-camera';
import React, { useEffect, useRef, useState } from 'react';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import { Alert, Button, Platform, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import ScreenWrapper from '../components/ScreenWrapper';
import { api } from '../lib/api';
import { useUser } from '../context/user-context';
import { useTranslation } from 'react-i18next';
import { useColorScheme } from '../hooks/use-color-scheme';
import { getAppColors } from '../lib/ui-theme';
import { requireOptionalNativeModule } from 'expo-modules-core';
import * as ImageManipulator from 'expo-image-manipulator';
import * as UPNG from 'upng-js';
import { toByteArray } from 'base64-js';

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView | null>(null);
  const [loading, setLoading] = useState(false);
  const [canCapture, setCanCapture] = useState(Platform.OS === 'web');
  const [torchOn, setTorchOn] = useState(false);
  const { t } = useTranslation();
  const [guideText, setGuideText] = useState(t('camera.checking'));
  const [liveBox, setLiveBox] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const [previewSize, setPreviewSize] = useState<{ width: number; height: number } | null>(null);
  const scanInFlightRef = useRef(false);
  const { user } = useUser();
  const colorScheme = useColorScheme();
  const c = getAppColors(colorScheme);

  const hasPermission = !!permission?.granted;

  useEffect(() => {
    if (!permission) {
      setGuideText(t('camera.checking'));
      return;
    }

    if (Platform.OS === 'web') {
      setCanCapture(true);
      setGuideText(t('camera.align'));
      return;
    }

    if (!hasPermission) {
      setCanCapture(false);
      setLiveBox(null);
      setGuideText(t('camera.tapAllow'));
      return;
    }

    let mounted = true;

    const tick = async () => {
      if (!mounted) return;
      if (loading) return;
      if (!cameraRef.current) return;
      if (scanInFlightRef.current) return;

      scanInFlightRef.current = true;
      try {
        // Low quality snapshot for fast CV gating.
        let frame: any = null;
        try {
          frame = await (cameraRef.current as any).takePictureAsync({ quality: 0.25, base64: false, skipProcessing: true });
        } catch {
          frame = await (cameraRef.current as any).takePictureAsync({ quality: 0.25, base64: false });
        }
        if (!frame?.uri) return;

        const formData = new FormData();
        formData.append('file', {
          uri: frame.uri,
          name: 'frame.jpg',
          type: 'image/jpeg',
        } as any);

        const url = `${api.defaults.baseURL}/detect-hand-live`;
        const res = await fetch(url, { method: 'POST', body: formData });
        if (!res.ok) return;
        const data = await res.json();

        const detected = !!data?.detected;
        const reason = data?.reason;
        const box = data?.box;
        if (mounted) {
          setCanCapture(detected);
          setLiveBox(detected && box ? box : null);

          if (detected) {
            setGuideText(t('camera.ready'));
          } else if (reason === 'move_closer') {
            setGuideText(t('camera.moveCloser'));
          } else {
            setGuideText(t('camera.align'));
          }
        }
      } catch {
        // Ignore transient camera/network errors during preview.
      } finally {
        scanInFlightRef.current = false;
      }
    };

    // Poll every ~0.5s for AR-style responsiveness.
    const id = setInterval(tick, 500);
    // Kick once quickly.
    tick();

    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [permission, hasPermission, loading, t]);

  const uriToBase64 = async (uri: string) => {
    const res = await fetch(uri);
    const blob = await res.blob();
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('Could not read image'));
      reader.onloadend = () => {
        const result = String(reader.result || '');
        const comma = result.indexOf(',');
        resolve(comma >= 0 ? result.slice(comma + 1) : result);
      };
      reader.readAsDataURL(blob);
    });
  };

  const uploadPalm = async (input: { uri?: string; base64?: string }) => {
    const name = user?.name?.trim() || 'Friend';
    const dob = user?.dob?.trim() || '2000-01-01';

    const analyzeOnDeviceAndSend = async (uri: string) => {
      // Convert to a small PNG (predictable decoding) and compute an edge-density feature.
      // Then send ONLY numbers to the backend.
      const manipulated = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 512 } }],
        { format: ImageManipulator.SaveFormat.PNG, base64: true }
      );

      const b64 = manipulated?.base64;
      if (!b64) throw new Error('Could not read image');

      const bytes = toByteArray(b64);
      const png = UPNG.decode(bytes.buffer);
      // toRGBA8 returns an array of frames; we only have 1 frame.
      const rgbaFrames = UPNG.toRGBA8(png);
      const rgba = rgbaFrames?.[0];
      if (!rgba) throw new Error('Could not decode image');

      const w = png.width | 0;
      const h = png.height | 0;
      if (!w || !h) throw new Error('Invalid image size');

      // Grayscale
      const gray = new Uint8Array(w * h);
      for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
        const r = rgba[p] as number;
        const g = rgba[p + 1] as number;
        const b = rgba[p + 2] as number;
        gray[i] = ((r * 30 + g * 59 + b * 11) / 100) | 0;
      }

      // Simple Sobel edge count (proxy for "line density")
      let edgeCount = 0;
      const thr = 80; // tuned for 512px inputs; adjust if needed

      for (let y = 1; y < h - 1; y++) {
        const row = y * w;
        for (let x = 1; x < w - 1; x++) {
          const i = row + x;

          const a00 = gray[i - w - 1];
          const a01 = gray[i - w];
          const a02 = gray[i - w + 1];
          const a10 = gray[i - 1];
          const a12 = gray[i + 1];
          const a20 = gray[i + w - 1];
          const a21 = gray[i + w];
          const a22 = gray[i + w + 1];

          const gx = -a00 + a02 - (a10 << 1) + (a12 << 1) - a20 + a22;
          const gy = -a00 - (a01 << 1) - a02 + a20 + (a21 << 1) + a22;
          const mag = Math.abs(gx) + Math.abs(gy);
          if (mag > thr) edgeCount++;
        }
      }

      const res = await fetch(`${api.defaults.baseURL}/analyze-palm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lineDensity: edgeCount, name, dob }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Analyze failed (${res.status})`);
      }
      return await res.json();
    };

    // Prefer on-device feature extraction on native.
    if (Platform.OS !== 'web' && input?.uri) {
      try {
        return await analyzeOnDeviceAndSend(input.uri);
      } catch {
        // If on-device CV fails, fall back to image upload.
      }

      const formData = new FormData();
      formData.append('file', {
        uri: input.uri,
        name: 'palm.jpg',
        type: 'image/jpeg',
      } as any);
      formData.append('name', name);
      formData.append('dob', dob);
      formData.append('overlay', 'false');

      const url = `${api.defaults.baseURL}/scan-palm`;
      const res = await fetch(url, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Scan failed (${res.status})`);
      }
      return await res.json();
    }

    let base64 = input?.base64;
    if (!base64 && Platform.OS === 'web' && input?.uri) {
      base64 = await uriToBase64(input.uri);
    }
    if (!base64) throw new Error('No image available to upload');

    const res = await api.post(
      '/upload-palm',
      {
        image: base64,
        name,
        dob,
      },
      {
        timeout: 60000,
        headers: { 'Content-Type': 'application/json' },
      }
    );
    return res?.data;
  };

  const showResultAlert = (data: any) => {
    if (typeof data?.result === 'string' && data.result.trim()) {
      Alert.alert(t('camera.analysis'), data.result.trim());
      return;
    }

    const traits: string[] = data?.traits ?? data?.analysis?.traits ?? [];
    const personality = data?.personality;
    const personalityTraits: string[] = personality?.traits ?? [];
    const topLabel: string | undefined = personality?.matches?.[0]?.label;
    const combinedTraits = Array.from(new Set([...(personalityTraits || []), ...(traits || [])]));

    if (combinedTraits.length) {
      const title = topLabel ? `AI Analysis — ${topLabel}` : t('camera.analysis');
      Alert.alert(title, combinedTraits.join('\n'));
    } else {
      Alert.alert(t('camera.success'), t('camera.uploaded'));
    }
  };

  const takePhoto = async () => {
    if (!cameraRef.current || loading || (!canCapture && Platform.OS !== 'web')) return;
    setLoading(true);

    try {
      // NOTE: CameraView exposes takePictureAsync via its ref in Expo SDKs that support it.
      // If your SDK complains about typing, we can type the ref as `any`.
      const photo: any = await (cameraRef.current as any).takePictureAsync({ base64: true, quality: 0.8 });
      if (!photo?.uri && !photo?.base64) throw new Error('No image returned from camera');

      const data = await uploadPalm({ uri: photo?.uri, base64: photo?.base64 });
      showResultAlert(data);
    } catch (e: any) {
      Alert.alert(t('camera.uploadFailed'), e?.message ?? t('camera.couldNotUpload'));
    } finally {
      setLoading(false);
    }
  };

  const getImagePicker = async () => {
    // Avoid crashing the app when running a dev-client build that
    // doesn't include expo-image-picker in the native binary.
    if (Platform.OS !== 'web') {
      const native = requireOptionalNativeModule?.('ExponentImagePicker');
      if (!native) return null;
    }

    try {
      return await import('expo-image-picker');
    } catch {
      return null;
    }
  };

  const pickFromGallery = async () => {
    if (loading) return;
    setLoading(true);

    try {
      const ImagePicker = await getImagePicker();
      if (!ImagePicker) {
        Alert.alert(t('camera.uploadFailed'), t('camera.galleryUnavailable'));
        return;
      }

      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        Alert.alert(t('camera.uploadFailed'), t('camera.allowPhotos'));
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.9,
        base64: Platform.OS === 'web',
        selectionLimit: 1,
      });

      if (result.canceled) return;
      const asset = result.assets?.[0];
      if (!asset?.uri) throw new Error('No image selected');

      const data = await uploadPalm({ uri: asset.uri, base64: (asset as any).base64 });
      showResultAlert(data);
    } catch (e: any) {
      Alert.alert(t('camera.uploadFailed'), e?.message ?? t('camera.couldNotUpload'));
    } finally {
      setLoading(false);
    }
  };

  return (
    !hasPermission ? (
      <ScreenWrapper>
        <Button title={t('camera.allow')} onPress={requestPermission} />
      </ScreenWrapper>
    ) : (
    <View style={{ flex: 1, backgroundColor: 'black' }}>
      <CameraView
        ref={cameraRef}
        style={{ flex: 1 }}
        facing="back"
        enableTorch={torchOn}
        onLayout={(e) => setPreviewSize({ width: e.nativeEvent.layout.width, height: e.nativeEvent.layout.height })}
      />

      <View style={styles.overlay}>
        <View style={[styles.betaPill, { backgroundColor: c.badgeBg, borderColor: c.badgeBorder }]}>
          <Text style={[styles.betaText, { color: c.badgeText }]}>{t('camera.beta')}</Text>
        </View>
        <View style={[styles.frame, { borderColor: c.primary }]} />
        {!!liveBox && !!previewSize && (
          <View
            pointerEvents="none"
            style={{
              position: 'absolute',
              borderWidth: 3,
              borderColor: c.primary2,
              borderRadius: 14,
              left: liveBox.x1 * previewSize.width,
              top: liveBox.y1 * previewSize.height,
              width: (liveBox.x2 - liveBox.x1) * previewSize.width,
              height: (liveBox.y2 - liveBox.y1) * previewSize.height,
            }}
          />
        )}
        <Text style={styles.text}>{guideText}</Text>
        <Text style={styles.subtext}>{t('camera.tips')}</Text>

        <View style={styles.controlsRow}>
          <TouchableOpacity
            style={[
              styles.smallBtn,
              { borderColor: c.accentBorderSoft, backgroundColor: 'rgba(0,0,0,0.25)' },
              torchOn && { borderColor: c.accentBorderStrong, backgroundColor: 'rgba(91,110,255,0.18)' },
              loading && { opacity: 0.55 },
            ]}
            onPress={() => setTorchOn((v) => !v)}
            disabled={loading}
          >
            <MaterialIcons name={torchOn ? 'flashlight-on' : 'flashlight-off'} size={22} color={'white'} />
            <Text style={styles.smallBtnText}>{t('camera.flash')}</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.captureBtn,
              { backgroundColor: c.primary },
              (loading || !canCapture) && { opacity: 0.55 },
            ]}
            onPress={takePhoto}
            disabled={loading || !canCapture}
          >
            <Text style={{ color: c.onPrimary, fontWeight: '800' }}>
              {loading ? t('camera.uploading') : canCapture ? t('camera.capture') : t('camera.hold')}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.smallBtn,
              { borderColor: c.accentBorderSoft, backgroundColor: 'rgba(0,0,0,0.25)' },
              loading && { opacity: 0.55 },
            ]}
            onPress={pickFromGallery}
            disabled={loading}
          >
            <MaterialIcons name="photo-library" size={22} color={'white'} />
            <Text style={styles.smallBtnText}>{t('camera.gallery')}</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
    )
  );
}

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    bottom: 50,
    width: '100%',
    alignItems: 'center',
  },
  betaPill: {
    borderWidth: 1,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
    marginBottom: 12,
  },
  betaText: {
    fontWeight: '800',
  },
  frame: {
    width: 220,
    height: 300,
    borderWidth: 3,
    borderRadius: 20,
  },
  text: {
    color: 'white',
    marginTop: 15,
    fontSize: 16,
    fontWeight: '700',
  },
  subtext: {
    color: 'rgba(255,255,255,0.82)',
    marginTop: 8,
    fontSize: 13,
  },
  controlsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 18,
    gap: 14,
  },
  smallBtn: {
    width: 70,
    height: 56,
    borderRadius: 16,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  smallBtnText: {
    color: 'white',
    fontSize: 11,
    marginTop: 2,
    fontWeight: '700',
  },
  captureBtn: {
    paddingVertical: 16,
    paddingHorizontal: 26,
    borderRadius: 999,
  },
});
