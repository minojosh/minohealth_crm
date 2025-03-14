/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_SPEECH_SERVICE_URL: process.env.SPEECH_SERVICE_URL,
    NEXT_PUBLIC_STT_SERVER_URL: process.env.STT_SERVER_URL
  }
};

module.exports = nextConfig;
