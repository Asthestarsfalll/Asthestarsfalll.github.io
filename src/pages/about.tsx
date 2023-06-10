import React, { useEffect, useState } from "react";
import Layout from "@theme/Layout";
import clsx from "clsx";
import arrayShuffle from "array-shuffle";
import BrowserOnly from '@docusaurus/BrowserOnly';

function About() {
  return (
    <Layout>
      <Friends />
      <BrowserOnly>
        {() => {
          useEffect(()=>{
            const Sakana = require('../plugins/sakana')
            //@ts-ignore
            Sakana.init({el: '.sakana-box', scale: .5, canSwitchCharacter: true} as any)
          },[])
          return <div className="sakana-box" style={{ height: 0 }}></div>
        }}
      </BrowserOnly>
      <p style={{ paddingLeft: '20px' }}>The list is random. try to refresh the page.</p>
    </Layout>
  );
}

interface FriendData {
  pic: string;
  name: string;
  intro: string;
  url: string;
  note: string;
}

function githubPic(name: string) {
  return `https://github.yuuza.net/${name}.png`;
}

var friendsData: FriendData[] = [
  {
    pic: githubPic("lideming"),
    name: "lideming",
    intro: "Building random things with Deno, Node and .NET Core.",
    url: "https://yuuza.net/",
    note: "佬，全栈/APEX/R6/CSGO，有些懒但实力强，导师又爱又恨。目前任职于ByteDance。",
  },
  {
    pic: githubPic("Therainisme"),
    name: "Therainisme",
    intro: "寄忆犹新",
    url: "https://blog.therainisme.com/",
    note: "佬（已保研）自称开源社区屎王, 他称鱼院（士）。目前主攻区块练存储技术，同时是网站前后端开发运维专家。",
  },
  {
    pic: githubPic("AndSonder"),
    name: "Sonder",
    intro: "life is but a span, I use python",
    url: "https://blog.keter.top/",
    note: "佬（已保研）。主攻方向是使用深度学习技术实现异常检测。",
  },
  {
    pic: githubPic("Zerorains"),
    name: "Zerorains",
    intro: "life is but a span, I use python",
    url: "https://blog.zerorains.top",
    note: "佬（已保研）。科协恶霸(x)。主攻方向是计算优化，同时熟悉深度学习技术。",
  },
  {
    pic: githubPic("PommesPeter"),
    name: "PommesPeter",
    intro: "Blessed with good gradient.",
    url: "https://memo.pommespeter.space/",
    note: "佬（已保研）。主攻方向是多模态场景图生成。",
  },
  {
    pic: githubPic("breezeshane"),
    name: "Breeze Shane",
    intro: "一个专注理论但学不懂学不会的锈钢废物，但是他很擅长产出Bug，可能是因为他体表有源石结晶分布，但也可能仅仅是因为他是Bug本体。",
    url: "https://breezeshane.github.io/",
    note: "一代传奇，手撸GAN的老单。",
  },
  {
    pic: githubPic("AndPuQing"),
    name: "PuQing",
    intro: "intro * new",
    url: "https://puqing.work",
    note: "梁老师感兴趣的不是程序，不是科技，不是摄影，而是能够表达自己的方式，却落得了专业写代码非常懂科技很会摄影的下场。",
  },
];

function Friends() {
  const [friends, setFriends] = useState<FriendData[]>(friendsData);
  useEffect(() => {
    setFriends(arrayShuffle(friends))
  }, []);
  const [current, setCurrent] = useState(0);
  const [previous, setPrevious] = useState(0);
  useEffect(() => {
    // After `current` change, set a 300ms timer making `previous = current` so the previous card will be removed.
    const timer = setTimeout(() => {
      setPrevious(current);
    }, 300);

    return () => {
      // Before `current` change to another value, remove (possibly not triggered) timer, and make `previous = current`.
      clearTimeout(timer);
      setPrevious(current);
    };
  }, [current]);
  return (
    <div className="friends" lang="zh-cn">
      <div style={{ position: "relative" }}>
        <div className="friend-columns">
          {/* Big card showing current selected */}
          <div className="friend-card-outer">
            {[
              previous != current && (
                <FriendCard key={previous} data={friends[previous]} fadeout />
              ),
              <FriendCard key={current} data={friends[current]} />,
            ]}
          </div>

          <div className="friend-list">
            {friends.map((x, i) => (
              <div
                key={x.name}
                className={clsx("friend-item", {
                  current: i == current,
                })}
                onClick={() => setCurrent(i)}
              >
                <img src={x.pic} alt="user profile photo" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function FriendCard(props: { data: FriendData; fadeout?: boolean }) {
  const { data, fadeout = false } = props;
  return (
    <div className={clsx("friend-card", { fadeout })}>
      <div className="card">
        <div className="card__image">
          <img
            src={data.pic}
            alt="User profile photo"
            title="User profile photo"
          />
        </div>
        <div className="card__body">
          <h2>{data.name}</h2>
          <p>
            <big>{data.intro}</big>
          </p>
          <p>
            <small>Comment : {data.note}</small>
          </p>
        </div>
        <div className="card__footer">
          <a href={data.url} className="button button--primary button--block">
            Visit
          </a>
        </div>
      </div>
    </div>
  );
}

export default About;
