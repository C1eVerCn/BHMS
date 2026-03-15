import React from 'react'
import { Typography } from 'antd'

const { Paragraph, Text, Title } = Typography

type HeroTone = 'teal' | 'amber' | 'rose' | 'slate'

interface HeroPill {
  label: string
  value: React.ReactNode
  tone?: HeroTone
}

interface PageHeroProps {
  kicker: string
  title: string
  description: React.ReactNode
  pills?: HeroPill[]
  aside?: React.ReactNode
  className?: string
}

const PageHero: React.FC<PageHeroProps> = ({ kicker, title, description, pills = [], aside, className }) => {
  return (
    <section className={['page-hero', className].filter(Boolean).join(' ')}>
      <div className="page-hero__main">
        <div className="page-hero__content">
          <Text className="page-hero__kicker">{kicker}</Text>
          <Title className="page-hero__title" level={2}>
            {title}
          </Title>
          <Paragraph className="page-hero__description">{description}</Paragraph>
        </div>

        {pills.length > 0 && (
          <div className="page-hero__meta">
            {pills.map((pill) => (
              <div key={pill.label} className={`page-pill page-pill--${pill.tone ?? 'slate'}`}>
                <span>{pill.label}</span>
                <strong>{pill.value}</strong>
              </div>
            ))}
          </div>
        )}
      </div>

      {aside ? <div className="page-hero__aside">{aside}</div> : null}
    </section>
  )
}

export default PageHero
