import type { Metadata } from 'next';
import './globals.css';

const isGitHubPages = process.env.GITHUB_ACTIONS === 'true';
const githubOwner = process.env.GITHUB_REPOSITORY_OWNER ?? 'CSProfKGD';
const repositoryName = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? 'eigenfaces-linear-algebra';
const siteOrigin = isGitHubPages
  ? `https://${githubOwner}.github.io/${repositoryName}`
  : 'https://eigenfaces-linear-algebra.csprofkgd.chatgpt.site';
const socialImage = isGitHubPages ? `${siteOrigin}/og.png` : '/og.png';

export const metadata: Metadata = {
  metadataBase: new URL(siteOrigin),
  title: 'Eigenfaces — Face It: It’s Just Linear Algebra',
  description:
    'An interactive visualization of a face reconstructed as an average plus weighted principal components.',
  openGraph: {
    title: 'Eigenfaces',
    description: 'Face It: It’s Just Linear Algebra',
    type: 'website',
    images: [
      {
        url: socialImage,
        width: 1200,
        height: 630,
        alt: 'Eigenfaces — Face It: It’s Just Linear Algebra',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Eigenfaces',
    description: 'Face It: It’s Just Linear Algebra',
    images: [socialImage],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
