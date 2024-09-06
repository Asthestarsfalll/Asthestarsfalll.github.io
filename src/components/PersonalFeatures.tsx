/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import React from "react";
import clsx from "clsx";
import styles from "./PersonalFeatures.module.css";

type FeatureItem = {
  title: string;
  image: string;
  description: JSX.Element;
  buttonLink?: string;
  buttonText?: string;
};

export const FeatureList: FeatureItem[] = [
  {
    title: "Me as CVer",
    image: "/img/illustrations/undraw_multitasking_re_ffpb.svg",
    description: (
      <>
        Focusing on Semantic Segmentation now. Pursuing Perfection. 
      </>
    ),
    buttonLink: "/docs/深度学习/",
    buttonText: "See my notes",
  },
  {
    title: "Me as Programmer",
    image: "/img/illustrations/undraw_programming_re_kg9v.svg",
    description: (
      <>
        Using Python & C++ & cuda. Pursuing Brevity and Elegance.
      </>
    ),
    buttonLink: "/blog",
    buttonText: "random blogs",
  },
  {
    title: "Me",
    image: "/img/illustrations/undraw_to_the_moon_re_q21i.png",
    description: (
      <>
        An rubbish undergraduate. Being obsessional about Customized keyboards & Music especially Post Rock.
      </>
    ),
    buttonLink: "/docs/meaningless/index",
    buttonText: "meaningless corner",
  },
];

export function Feature({
  title,
  image,
  description,
  buttonLink,
  buttonText,
}: FeatureItem) {
  return (
    <div className={clsx("col col--4")} style={{display: 'flex', flexDirection: 'column', justifyContent: 'space-between'}}>
      <div className="text--center">
        <img className={styles.featureSvg} alt={title} src={image} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
      {buttonLink ? (
        <div style={{ textAlign: "center", }}>
          <a
            href={buttonLink}
            className="button button--primary button--outline"
          >
            {buttonText}
          </a>
        </div>
      ) : null}
    </div>
  );
}

export function PersonalFeatures(): JSX.Element {
  return (
    <section className={clsx(styles.features,"hero")}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
