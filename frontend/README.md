# GraphRAG-MultiHop Frontend

Next.js 14 + TypeScript + Tailwind CSS frontend for GraphRAG-MultiHop.

## Features

- 🎨 Modern, responsive UI with Tailwind CSS
- ⚡ Real-time query interface with streaming
- 📊 Results visualization with citations
- 🔧 Configurable retrieval methods (Vector, Graph, Hybrid)
- 📱 Mobile-first responsive design
- 🌙 Dark mode support
- 🚀 Vercel-ready deployment

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Environment Variables

Create `.env.local`:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

## Vercel Deployment

1. Push to GitHub
2. Import in Vercel
2. Add `NEXT_PUBLIC_BACKEND_URL` environment variable
3. Deploy!

## Architecture

```
src/
├── app/                    # Next.js App Router
│   ├── api/               # API routes (proxy to FastAPI)
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Main page
├── components/            # React components
│   ├── QueryInterface.tsx # Question input
│   ├── ResultsPanel.tsx   # Answer display
│   └── Sidebar.tsx        # Settings panel
└── lib/
    └── utils.ts           # Utility functions
```

## Backend Integration

The frontend proxies API calls to the FastAPI backend at `NEXT_PUBLIC_BACKEND_URL`.

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Utilities**: clsx, tailwind-merge
- **HTTP**: Native fetch (via API routes)