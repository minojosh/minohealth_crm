/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_SPEECH_SERVER_URL: process.env.SPEECH_SERVER_URL
  }
};

module.exports = nextConfig;
