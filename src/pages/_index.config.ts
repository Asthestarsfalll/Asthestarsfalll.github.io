interface TypicalText {
  text: string;
  delay: number;
}

const delay = 3000;

export const subtitles_and_delays: TypicalText[] = [
  { text: "我们登上并非我们所选择的舞台,", delay: delay },
  { text: "演出并非我们所选择的剧本。", delay: delay },
  { text: "我们太有限了，", delay: delay },
  { text: "我们只能做我们觉得是对的事，", delay: delay },
  { text: "然后接受它的事与愿违。", delay: delay },
  { text: "人所有的拖沓都是代表他并非真正热爱。", delay: delay },
  { text: "人所有的一切都是骗局，生命完全是偶尔。", delay: delay },

  // { text: "Rubbish CVer", delay: 1000 },
  // { text: "Nihilist", delay: 1000 },
  // { text: "INTP", delay: 1000 },
  // { text: "Individualist", delay: 1000 },
];

//should be in [0,12]
export const title_width: number = 7;

export const illustrations: string[] = [
  "/img/illustrations/undraw_coding_re_iv62.svg",
  "/img/illustrations/undraw_launch_day_4e04.svg",
  "/img/illustrations/undraw_developer_activity_re_39tg.svg",
  "/img/illustrations/undraw_floating_re_xtcj.svg",
  "/img/illustrations/undraw_outer_space_re_u9vd.svg",
  "/img/illustrations/undraw_exploring_re_grb8.svg",
  "/img/illustrations/undraw_stars_re_6je7.svg",
  "/img/illustrations/undraw_compose_music_re_wpiw.svg",
  "/img/illustrations/undraw_people_re_8spw.svg",
];
