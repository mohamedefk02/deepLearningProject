import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8010';

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 120000,
});

export const autocomplete = ({ input, method, length, temperature, beamWidth }) =>
  client.post('/api/autocomplete', {
    input,
    method,
    length,
    temperature,
    beam_width: beamWidth,
  });

export const translate = (input) =>
  client.post('/api/translate', { input });

export const classify = (input) =>
  client.post('/api/classify', { input });

export const health = () => client.get('/health');

