/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_SPEECH_SERVICE_URL: process.env.SPEECH_SERVICE_URL
  }
};

module.exports = nextConfig;
