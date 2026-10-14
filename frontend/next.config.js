/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://localhost:8000/:path*',
      },
    ];
  },
  images: {
    domains: ['localhost'],
  },
};

module.exports = nextConfig;