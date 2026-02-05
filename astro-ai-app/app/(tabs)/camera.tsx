import { CameraView, useCameraPermissions } from 'expo-camera';
import React, { useEffect, useRef, useState } from 'react';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import { Alert, Button, Platform, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import ScreenWrapper from '../../components/ScreenWrapper';
import { api } from '../../lib/api';
import { useUser } from '../../context/user-context';
import { useTranslation } from 'react-i18next';
import { useColorScheme } from '../../hooks/use-color-scheme';
import { getAppColors } from '../../lib/ui-theme';
import * as ImageManipulator from 'expo-image-manipulator';
import * as UPNG from 'upng-js';
import { toByteArray } from 'base64-js';
import { appendLocalPalmHistory } from '../../lib/palm-history';

type PalmLineClass = 'heart' | 'head' | 'life' | 'other';

type PalmFeaturesV1 = {
  version: 1;
  image: { width: number; height: number };
  roi: {
    x: number;
    y: number;
    width: number;
    height: number;
    fillRatio: number;
  };
  line: {
    pixelCount: number;
    pixelDensity: number;
  };
  segments: {
    totalCount: number;
    totalLength: number;
    heart: { count: number; length: number; meanAngleDeg: number };
    head: { count: number; length: number; meanAngleDeg: number };
    life: { count: number; length: number; meanAngleDeg: number };
  };
};

const tryRequireFastOpenCV = (): null | any => {
  try {
    // NOTE: Native module. Requires a dev-client/EAS build.
    // Keep require() inside a function so Expo Go/web don't hard-crash.
    return require('react-native-fast-opencv');
  } catch {
    return null;
  }
};

const classifySegment = (midX: number, midY: number, angleDegAbs: number): PalmLineClass => {
  // Heuristic, normalized to ROI coordinates:
  // - Heart line: upper palm, roughly horizontal
  // - Head line: middle palm, roughly horizontal
  // - Life line: near either side (thumb side differs per hand), more diagonal/vertical
  if (midY < 0.35 && angleDegAbs < 45) return 'heart';
  if (midY < 0.70 && angleDegAbs < 45) return 'head';
  const sideProximity = Math.min(midX, 1 - midX);
  if (sideProximity < 0.25 && angleDegAbs > 30) return 'life';
  return 'other';
};

const safeMeanAngleDeg = (anglesDegAbs: number[]): number => {
  if (!anglesDegAbs.length) return 0;
  const s = anglesDegAbs.reduce((a, b) => a + b, 0);
  return s / anglesDegAbs.length;
};

const extractPalmFeaturesWithFastOpenCV = (params: {
  base64Png: string;
  width: number;
  height: number;
}): PalmFeaturesV1 => {
  const mod = tryRequireFastOpenCV();
  if (!mod?.OpenCV) throw new Error('FastOpenCV not available');

  const {
    OpenCV,
    ObjectType,
    DataTypes,
    ColorConversionCodes,
    MorphShapes,
    MorphTypes,
    RetrievalModes,
    ContourApproximationModes,
    AdaptiveThresholdTypes,
    ThresholdTypes,
  } = mod;

  const { base64Png, width, height } = params;

  // Keep everything best-effort; always release buffers.
  try {
    const src = OpenCV.base64ToMat(base64Png);

    const hsv = OpenCV.createObject(ObjectType.Mat, height, width, DataTypes.CV_8UC3);
    OpenCV.invoke('cvtColor', src, hsv, ColorConversionCodes.COLOR_BGR2HSV);

    // HSV skin-ish mask (broad, works ok across lighting)
    const lower = OpenCV.createObject(ObjectType.Scalar, 0, 25, 40);
    const upper = OpenCV.createObject(ObjectType.Scalar, 30, 255, 255);
    const skinMask = OpenCV.createObject(ObjectType.Mat, height, width, DataTypes.CV_8UC1);
    OpenCV.invoke('inRange', hsv, lower, upper, skinMask);

    // Clean mask
    const kSizeClose = OpenCV.createObject(ObjectType.Size, 9, 9);
    const kClose = OpenCV.invoke('getStructuringElement', MorphShapes.MORPH_ELLIPSE, kSizeClose);
    const maskClosed = OpenCV.createObject(ObjectType.Mat, height, width, DataTypes.CV_8UC1);
    OpenCV.invoke('morphologyEx', skinMask, maskClosed, MorphTypes.MORPH_CLOSE, kClose);

    const contours = OpenCV.createObject(ObjectType.MatVector);
    OpenCV.invoke('findContours', maskClosed, contours, RetrievalModes.RETR_EXTERNAL, ContourApproximationModes.CHAIN_APPROX_SIMPLE);

    const contourMeta = OpenCV.toJSValue(contours);
    const contourCount: number = Array.isArray(contourMeta?.array) ? contourMeta.array.length : 0;
    if (!contourCount) {
      throw new Error('No palm contour found');
    }

    let bestIdx = 0;
    let bestArea = -1;
    for (let i = 0; i < contourCount; i++) {
      const contour = OpenCV.copyObjectFromVector(contours, i);
      const a = OpenCV.invoke('contourArea', contour)?.value ?? 0;
      if (a > bestArea) {
        bestArea = a;
        bestIdx = i;
      }
    }
    const bestContour = OpenCV.copyObjectFromVector(contours, bestIdx);
    const roiRect = OpenCV.invoke('boundingRect', bestContour);
    const roiX = Math.max(0, roiRect?.x ?? 0);
    const roiY = Math.max(0, roiRect?.y ?? 0);
    const roiW = Math.max(1, roiRect?.width ?? width);
    const roiH = Math.max(1, roiRect?.height ?? height);

    const roi = OpenCV.createObject(ObjectType.Mat, roiH, roiW, DataTypes.CV_8UC3);
    OpenCV.invoke('crop', src, roi, roiRect);
    const roiMask = OpenCV.createObject(ObjectType.Mat, roiH, roiW, DataTypes.CV_8UC1);
    OpenCV.invoke('crop', maskClosed, roiMask, roiRect);

    const roiMaskPixels = OpenCV.invoke('countNonZero', roiMask)?.value ?? 0;
    const roiArea = roiW * roiH;
    const fillRatio = roiArea > 0 ? roiMaskPixels / roiArea : 0;

    // Line enhancement + binarization
    const gray = OpenCV.createObject(ObjectType.Mat, roiH, roiW, DataTypes.CV_8UC1);
    OpenCV.invoke('cvtColor', roi, gray, ColorConversionCodes.COLOR_BGR2GRAY);

    const blur = OpenCV.createObject(ObjectType.Mat, roiH, roiW, DataTypes.CV_8UC1);
    const blurK = OpenCV.createObject(ObjectType.Size, 5, 5);
    OpenCV.invoke('GaussianBlur', gray, blur, blurK, 0);

    const bin = OpenCV.createObject(ObjectType.Mat, roiH, roiW, DataTypes.CV_8UC1);
    OpenCV.invoke(
      'adaptiveThreshold',
      blur,
      bin,
      255,
      AdaptiveThresholdTypes.ADAPTIVE_THRESH_GAUSSIAN_C,
      ThresholdTypes.THRESH_BINARY_INV,
      15,
      2
    );

    // Keep only within ROI mask
    const binMasked = OpenCV.createObject(ObjectType.Mat, roiH, roiW, DataTypes.CV_8UC1);
    OpenCV.invoke('bitwise_and', bin, bin, binMasked, roiMask);

    const linePixelCount = OpenCV.invoke('countNonZero', binMasked)?.value ?? 0;
    const linePixelDensity = roiMaskPixels > 0 ? linePixelCount / roiMaskPixels : 0;

    // Edges + Hough segments
    const edges = OpenCV.createObject(ObjectType.Mat, roiH, roiW, DataTypes.CV_8UC1);
    OpenCV.invoke('Canny', blur, edges, 60, 160);

    const edgesMasked = OpenCV.createObject(ObjectType.Mat, roiH, roiW, DataTypes.CV_8UC1);
    OpenCV.invoke('bitwise_and', edges, edges, edgesMasked, roiMask);

    const lines = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_32SC4);
    OpenCV.invoke('HoughLinesP', edgesMasked, lines, 1, Math.PI / 180, 45);

    const linesBuf = OpenCV.matToBuffer(lines, 'int32');
    const buf: Int32Array | undefined = linesBuf?.buffer;
    const channels = Number(linesBuf?.channels ?? 0);
    const segCount = buf && channels === 4 ? Math.floor(buf.length / 4) : 0;

    let totalLength = 0;
    const byClass: Record<Exclude<PalmLineClass, 'other'>, { count: number; length: number; angles: number[] }> = {
      heart: { count: 0, length: 0, angles: [] },
      head: { count: 0, length: 0, angles: [] },
      life: { count: 0, length: 0, angles: [] },
    };

    const minLen = Math.max(10, Math.floor(Math.min(roiW, roiH) * 0.12));

    for (let i = 0; i < segCount; i++) {
      const x1 = buf![i * 4 + 0];
      const y1 = buf![i * 4 + 1];
      const x2 = buf![i * 4 + 2];
      const y2 = buf![i * 4 + 3];

      const dx = x2 - x1;
      const dy = y2 - y1;
      const len = Math.hypot(dx, dy);
      if (!Number.isFinite(len) || len < minLen) continue;

      const midX = (x1 + x2) / 2 / roiW;
      const midY = (y1 + y2) / 2 / roiH;
      const angleDegAbs = Math.abs((Math.atan2(dy, dx) * 180) / Math.PI);

      const cls = classifySegment(midX, midY, angleDegAbs);
      totalLength += len;

      if (cls === 'heart' || cls === 'head' || cls === 'life') {
        byClass[cls].count += 1;
        byClass[cls].length += len;
        byClass[cls].angles.push(angleDegAbs);
      }
    }

    return {
      version: 1,
      image: { width, height },
      roi: {
        x: roiX,
        y: roiY,
        width: roiW,
        height: roiH,
        fillRatio,
      },
      line: {
        pixelCount: linePixelCount,
        pixelDensity: linePixelDensity,
      },
      segments: {
        totalCount: segCount,
        totalLength,
        heart: {
          count: byClass.heart.count,
          length: byClass.heart.length,
          meanAngleDeg: safeMeanAngleDeg(byClass.heart.angles),
        },
        head: {
          count: byClass.head.count,
          length: byClass.head.length,
          meanAngleDeg: safeMeanAngleDeg(byClass.head.angles),
        },
        life: {
          count: byClass.life.count,
          length: byClass.life.length,
          meanAngleDeg: safeMeanAngleDeg(byClass.life.angles),
        },
      },
    };
  } finally {
    try {
      mod.OpenCV.clearBuffers();
    } catch {
      // ignore
    }
  }
};

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

  const uploadPalm = async (input: { uri?: string; base64?: string }) => {
    const name = user?.name?.trim() || 'Friend';
    const dob = user?.dob?.trim() || '2000-01-01';

    const analyzeOnDeviceAndSend = async (uri: string) => {
      // Convert to a small PNG (predictable decoding).
      // Then extract features on-device and send ONLY numbers to the backend.
      const manipulated = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 512 } }],
        { format: ImageManipulator.SaveFormat.PNG, base64: true }
      );

      const b64 = manipulated?.base64;
      if (!b64) throw new Error('Could not read image');

      // 1) Try real ROI segmentation + line segments via native OpenCV.
      // 2) Fallback to JS edge-density proxy if native module isn't in the build.
      let features: PalmFeaturesV1 | null = null;
      try {
        features = extractPalmFeaturesWithFastOpenCV({
          base64Png: b64,
          width: manipulated.width ?? 512,
          height: manipulated.height ?? 512,
        });
      } catch {
        features = null;
      }

      // Legacy numeric: edgeCount for backward compatibility & keepalive.
      let edgeCount = 0;
      if (!features) {
        const bytes = toByteArray(b64);
        const png = UPNG.decode(bytes.buffer);
        const rgbaFrames = UPNG.toRGBA8(png);
        const rgba = rgbaFrames?.[0];
        if (!rgba) throw new Error('Could not decode image');

        const w = png.width | 0;
        const h = png.height | 0;
        if (!w || !h) throw new Error('Invalid image size');

        const gray = new Uint8Array(w * h);
        for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
          const r = rgba[p] as number;
          const g = rgba[p + 1] as number;
          const b = rgba[p + 2] as number;
          gray[i] = ((r * 30 + g * 59 + b * 11) / 100) | 0;
        }

        let count = 0;
        const thr = 80;
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
            if (mag > thr) count++;
          }
        }
        edgeCount = count;
      } else {
        // When OpenCV features are present, keep a stable numeric proxy too.
        const roiArea = Math.max(1, features.roi.width * features.roi.height);
        edgeCount = Math.round(features.line.pixelCount * (512 * 512) / roiArea);
      }

      const res = await fetch(`${api.defaults.baseURL}/analyze-palm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          dob,
          lineDensity: edgeCount,
          features,
        }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Analyze failed (${res.status})`);
      }
      return await res.json();
    };

    if (!input?.uri) throw new Error('No image available');
    try {
      return await analyzeOnDeviceAndSend(input.uri);
    } catch {
      Alert.alert(t('camera.uploadFailed'), 'Lighting too low. Try again in brighter light.');
      return { result: '' };
    }
  };

  const showResultAlert = (data: any): string => {
    if (typeof data?.result === 'string' && data.result.trim()) {
      const msg = data.result.trim();
      Alert.alert(t('camera.analysis'), msg);
      return msg;
    }

    const traits: string[] = data?.traits ?? data?.analysis?.traits ?? [];
    const personality = data?.personality;
    const personalityTraits: string[] = personality?.traits ?? [];
    const topLabel: string | undefined = personality?.matches?.[0]?.label;
    const combinedTraits = Array.from(new Set([...(personalityTraits || []), ...(traits || [])]));

    if (combinedTraits.length) {
      const title = topLabel ? `AI Analysis — ${topLabel}` : t('camera.analysis');
      const msg = combinedTraits.join('\n');
      Alert.alert(title, msg);
      return msg;
    }

    const fallback = t('camera.uploaded');
    Alert.alert(t('camera.success'), fallback);
    return fallback;
  };

  const takePhoto = async () => {
    if (!cameraRef.current || loading || (!canCapture && Platform.OS !== 'web')) return;
    setLoading(true);

    const name = user?.name?.trim() || 'Friend';
    const dob = user?.dob?.trim() || '2000-01-01';

    try {
      const photo: any = await (cameraRef.current as any).takePictureAsync({ base64: true, quality: 0.8 });
      if (!photo?.uri && !photo?.base64) throw new Error('No image returned from camera');

      const data = await uploadPalm({ uri: photo?.uri, base64: photo?.base64 });
      const msg = showResultAlert(data);
      if (photo?.uri && name && dob && msg) {
        try {
          await appendLocalPalmHistory(name, dob, { imageUri: photo.uri, message: msg });
        } catch {
          // ignore
        }
      }
    } catch (e: any) {
      Alert.alert(t('camera.uploadFailed'), e?.message ?? t('camera.couldNotUpload'));
    } finally {
      setLoading(false);
    }
  };

  const getImagePicker = async () => {
    try {
      return await import('expo-image-picker');
    } catch {
      return null;
    }
  };

  const pickFromGallery = async () => {
    if (loading) return;
    setLoading(true);

    const name = user?.name?.trim() || 'Friend';
    const dob = user?.dob?.trim() || '2000-01-01';

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
      const msg = showResultAlert(data);
      if (asset?.uri && name && dob && msg) {
        try {
          await appendLocalPalmHistory(name, dob, { imageUri: asset.uri, message: msg });
        } catch {
          // ignore
        }
      }
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
