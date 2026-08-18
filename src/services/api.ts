import { createClient } from '@supabase/supabase-js';

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

export const supabase = createClient(
  supabaseUrl || 'https://placeholder.supabase.co',
  supabaseAnonKey || 'placeholder'
);

export const isBackendConfigured = true;

export type Prediction = {
  id?: number;
  image_name: string;
  image_path?: string;
  image_url?: string;
  species: string;
  confidence: number;
  created_at?: string;
};

export type Statistics = {
  total_predictions: number;
  unique_species: number;
  average_confidence: number;
  most_recognized_species: string | null;
  species_distribution: { species: string; count: number }[];
  species_percentage: { species: string; percentage: number }[];
  recognition_trend: { date: string; count: number }[];
};

export type HealthStatus = 'connected' | 'offline' | 'checking';

let activeBackend: 'fastapi' | 'supabase' | 'none' = 'none';

// ---- TensorFlow.js MobileNet (lazy-loaded fallback if needed) ----
let mobilenetModel: Awaited<ReturnType<typeof loadMobilenet>> | null = null;
let modelLoading: Promise<Awaited<ReturnType<typeof loadMobilenet>>> | null = null;

async function loadMobilenet() {
  const tf = await import('@tensorflow/tfjs');
  await tf.ready();
  const mobilenet = await import('@tensorflow-models/mobilenet');
  return mobilenet.load({ version: 2, alpha: 1.0 });
}

async function getMobilenet() {
  if (mobilenetModel) return mobilenetModel;
  if (!modelLoading) modelLoading = loadMobilenet();
  mobilenetModel = await modelLoading;
  return mobilenetModel;
}

// ---- Health check ----
export async function checkHealth(): Promise<boolean> {
  // 1. Try FastAPI Python Backend
  try {
    const res = await fetch(`${apiBaseUrl}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      activeBackend = 'fastapi';
      return true;
    }
  } catch {
    // FastAPI not reachable
  }

  // 2. Try Supabase if configured
  if (supabaseUrl && supabaseAnonKey && supabaseUrl !== 'https://placeholder.supabase.co') {
    try {
      const { error } = await supabase.from('predictions').select('id').limit(1).maybeSingle();
      if (!error) {
        activeBackend = 'supabase';
        return true;
      }
    } catch {
      // Supabase failed
    }
  }

  activeBackend = 'none';
  return false;
}

// ---- Prediction: run via FastAPI or Supabase/MobileNet ----
export async function predictImage(file: File): Promise<Prediction> {
  await checkHealth();

  if (activeBackend === 'fastapi') {
    const formData = new FormData();
    formData.append('image', file);
    const res = await fetch(`${apiBaseUrl}/predict`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: 'Failed to analyze image' }));
      throw new Error(errData.detail || 'Unable to recognize the image. Please try again.');
    }
    const data = await res.json();
    const fullImageUrl = data.image_url
      ? (data.image_url.startsWith('http') ? data.image_url : `${apiBaseUrl}${data.image_url}`)
      : '';
    return {
      id: data.id,
      image_name: data.image_name,
      image_path: data.image_path,
      image_url: fullImageUrl,
      species: data.species,
      confidence: data.confidence,
      created_at: data.created_at,
    };
  }

  if (activeBackend === 'supabase') {
    const model = await getMobilenet();
    const imgEl = await fileToImageElement(file);
    const rawPredictions = await model.classify(imgEl, 5);

    if (!rawPredictions || rawPredictions.length === 0) {
      throw new Error('Unable to recognize the image. Please try again.');
    }

    const top = rawPredictions[0];
    const species = top.className.split(',')[0].trim();
    const confidence = Math.round(top.probability * 10000) / 100;

    const ext = file.name.split('.').pop() || 'jpg';
    const fileName = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}.${ext}`;
    const { error: uploadError } = await supabase.storage
      .from('predictions')
      .upload(fileName, file, { contentType: file.type });

    let imagePath = '';
    if (!uploadError) {
      imagePath = supabase.storage.from('predictions').getPublicUrl(fileName).data.publicUrl;
    }

    const { data, error } = await supabase
      .from('predictions')
      .insert({
        image_name: file.name,
        image_path: imagePath,
        species,
        confidence,
      })
      .select()
      .single();

    if (error) throw error;

    return {
      id: data.id,
      image_name: data.image_name,
      image_path: data.image_path,
      image_url: data.image_path,
      species: data.species,
      confidence: data.confidence,
      created_at: data.created_at,
    };
  }

  throw new Error('Backend server is offline. Please make sure the Python FastAPI server is running on http://localhost:8000.');
}

// ---- History: list all predictions ----
export async function getPredictions(): Promise<Prediction[]> {
  await checkHealth();

  if (activeBackend === 'fastapi') {
    const res = await fetch(`${apiBaseUrl}/predictions`);
    if (!res.ok) throw new Error('Failed to fetch prediction history.');
    const rows = await res.json();
    return (rows || []).map((r: any) => ({
      id: r.id,
      image_name: r.image_name,
      image_path: r.image_path,
      image_url: r.image_url
        ? (r.image_url.startsWith('http') ? r.image_url : `${apiBaseUrl}${r.image_url}`)
        : '',
      species: r.species,
      confidence: r.confidence,
      created_at: r.created_at,
    }));
  }

  if (activeBackend === 'supabase') {
    const { data, error } = await supabase
      .from('predictions')
      .select('*')
      .order('id', { ascending: false });

    if (error) throw error;

    return (data || []).map((r) => ({
      id: r.id,
      image_name: r.image_name,
      image_path: r.image_path,
      image_url: r.image_path,
      species: r.species,
      confidence: r.confidence,
      created_at: r.created_at,
    }));
  }

  return [];
}

// ---- Get single prediction ----
export async function getPredictionById(id: number): Promise<Prediction> {
  await checkHealth();

  if (activeBackend === 'fastapi') {
    const res = await fetch(`${apiBaseUrl}/predictions/${id}`);
    if (!res.ok) throw new Error('Prediction not found.');
    const r = await res.json();
    return {
      id: r.id,
      image_name: r.image_name,
      image_path: r.image_path,
      image_url: r.image_url
        ? (r.image_url.startsWith('http') ? r.image_url : `${apiBaseUrl}${r.image_url}`)
        : '',
      species: r.species,
      confidence: r.confidence,
      created_at: r.created_at,
    };
  }

  if (activeBackend === 'supabase') {
    const { data, error } = await supabase
      .from('predictions')
      .select('*')
      .eq('id', id)
      .maybeSingle();

    if (error) throw error;
    if (!data) throw new Error('Prediction not found.');

    return {
      id: data.id,
      image_name: data.image_name,
      image_path: data.image_path,
      image_url: data.image_path,
      species: data.species,
      confidence: data.confidence,
      created_at: data.created_at,
    };
  }

  throw new Error('Backend server is offline.');
}

// ---- Statistics ----
export async function getStatistics(): Promise<Statistics> {
  await checkHealth();

  if (activeBackend === 'fastapi') {
    const res = await fetch(`${apiBaseUrl}/statistics`);
    if (!res.ok) throw new Error('Failed to fetch statistics.');
    return await res.json();
  }

  if (activeBackend === 'supabase') {
    const { data, error } = await supabase
      .from('predictions')
      .select('species, confidence, created_at');

    if (error) throw error;

    const rows = data || [];
    const total = rows.length;
    const speciesCounts: Record<string, number> = {};
    const confidences: number[] = [];
    const trend: Record<string, number> = {};

    for (const r of rows) {
      speciesCounts[r.species] = (speciesCounts[r.species] || 0) + 1;
      confidences.push(r.confidence);
      const day = (r.created_at || '').slice(0, 10);
      if (day) trend[day] = (trend[day] || 0) + 1;
    }

    const uniqueSpecies = Object.keys(speciesCounts).length;
    const avgConf = confidences.length ? confidences.reduce((a, b) => a + b, 0) / confidences.length : 0;
    const mostRecognized = Object.entries(speciesCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;

    const distribution = Object.entries(speciesCounts)
      .map(([species, count]) => ({ species, count }))
      .sort((a, b) => b.count - a.count);

    const percentage = distribution.map((d) => ({
      species: d.species,
      percentage: total ? Math.round((d.count / total) * 10000) / 100 : 0,
    }));

    const trendList = Object.entries(trend)
      .map(([date, count]) => ({ date, count }))
      .sort((a, b) => a.date.localeCompare(b.date));

    return {
      total_predictions: total,
      unique_species: uniqueSpecies,
      average_confidence: Math.round(avgConf * 100) / 100,
      most_recognized_species: mostRecognized,
      species_distribution: distribution,
      species_percentage: percentage,
      recognition_trend: trendList,
    };
  }

  return {
    total_predictions: 0,
    unique_species: 0,
    average_confidence: 0,
    most_recognized_species: null,
    species_distribution: [],
    species_percentage: [],
    recognition_trend: [],
  };
}

// ---- Delete prediction ----
export async function deletePrediction(id: number): Promise<void> {
  await checkHealth();

  if (activeBackend === 'fastapi') {
    const res = await fetch(`${apiBaseUrl}/predictions/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete prediction.');
    return;
  }

  if (activeBackend === 'supabase') {
    const { error } = await supabase.from('predictions').delete().eq('id', id);
    if (error) throw error;
    return;
  }

  throw new Error('Backend server is offline.');
}

// ---- Friendly error messages ----
export function friendlyError(err: unknown): string {
  if (err instanceof Error) {
    if (err.message.includes('Failed to fetch') || err.message.includes('network')) {
      return 'Unable to connect to the backend server. Please make sure the Python FastAPI server is running on http://localhost:8000.';
    }
    return err.message || 'Unable to recognize the image. Please try again.';
  }
  return 'Unable to recognize the image. Please try again.';
}

function fileToImageElement(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Invalid image: could not load for analysis.'));
    };
    img.src = url;
  });
}

