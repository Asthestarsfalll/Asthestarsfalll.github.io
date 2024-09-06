import React, { useEffect, useState } from "react";
import { ResponsivePie } from '@nivo/pie';
import { ResponsiveBar } from "@nivo/bar";
import { WakatimeData, useWakatimeData } from "../hooks/useWakatimeData";
import clsx from 'clsx'

interface Props {
    data: WakatimeData
}

export function WakatimeFeatures(props: Props): JSX.Element {
    return (
        <section>
            <div className={clsx("container","hero")}>
                <div className="row" style={{width:'inherit'}}>
                    <div className="col col--6" style={{ height: '400px' }}><WakaLangPieChart data={props.data.languages} /></div>
                    <div className="col col--6" style={{ height: '400px' }}><WakaEdtrBarChart data={props.data.editors} /></div>
                </div>
            </div>
        </section>
    );
}

export function WakatimeLanguages(props: Props) {
    return (
        <div className="col" style={{ height: '400px'}}><WakaLangPieChart data={props.data.languages} /></div>
    )
}

export function WakatimeEditors(props: Props) {
    return (
        <div className="col" style={{ height: '400px' }}><WakaEdtrBarChart data={props.data.editors} /></div>
    )
}

const WakaLangPieChart = ({ data /* see data tab */ }) => (
    <ResponsivePie
        data={data}
        theme={{
            tooltip: {
                basic: {
                    color: '#000000'
                }
            }
        }}
        margin={{ top: 40, right: 80, bottom: 80, left: 80 }}
        innerRadius={0.35}
        padAngle={1}
        cornerRadius={5}
        activeInnerRadiusOffset={10}
        activeOuterRadiusOffset={10}
        colors={{ scheme: 'reds' }}
        borderWidth={1}
        borderColor={{ from: 'color', modifiers: [['darker', 0.2]] }}
        arcLinkLabelsSkipAngle={10}
        arcLinkLabelsTextColor="#999999"
        arcLinkLabelsThickness={2}
        arcLinkLabelsColor={{ from: 'color' }}
        arcLabelsSkipAngle={10}
        arcLabelsTextColor={{ from: 'color', modifiers: [['darker', 2]] }}
        defs={[
            {
                id: 'dots',
                type: 'patternDots',
                background: 'inherit',
                color: 'rgba(255, 255, 255, 0.3)',
                size: 4,
                padding: 1,
                stagger: true
            },
            {
                id: 'lines',
                type: 'patternLines',
                background: 'inherit',
                color: 'rgba(255, 255, 255, 0.3)',
                rotation: -45,
                lineWidth: 6,
                spacing: 10
            }
        ]}
        fill={[
            {
                match: {
                    id: 'ruby'
                },
                id: 'dots'
            },
            {
                match: {
                    id: 'c'
                },
                id: 'dots'
            },
            {
                match: {
                    id: 'go'
                },
                id: 'dots'
            },
            {
                match: {
                    id: 'python'
                },
                id: 'dots'
            },
            {
                match: {
                    id: 'scala'
                },
                id: 'lines'
            },
            {
                match: {
                    id: 'lisp'
                },
                id: 'lines'
            },
            {
                match: {
                    id: 'elixir'
                },
                id: 'lines'
            },
            {
                match: {
                    id: 'javascript'
                },
                id: 'lines'
            }
        ]}
        motionConfig="wobbly"
        legends={[]}
    />
);

const WakaEdtrBarChart = ({ data /* see data tab */ }) => (
    <ResponsiveBar
        data={data}
        theme={{
            textColor: '#999999',
            tooltip: {
                basic: {
                    color: '#000000'
                }
            }
        }}
        keys={['value']}
        indexBy="name"
        margin={{ top: 50, right: 50, bottom: 50, left: 50 }}
        groupMode="grouped"
        valueScale={{ type: 'linear' }}
        indexScale={{ type: 'band', round: true }}
        padding={0.3}
        innerPadding={10}
        colors={{ scheme: 'reds' }}
        colorBy="indexValue"
        borderRadius={3}
        borderWidth={1}
        borderColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
        defs={[
            {
                id: 'dots',
                type: 'patternDots',
                background: 'inherit',
                color: '#38bcb2',
                size: 4,
                padding: 1,
                stagger: true
            },
            {
                id: 'lines',
                type: 'patternLines',
                background: 'inherit',
                color: '#eed312',
                rotation: -45,
                lineWidth: 6,
                spacing: 10
            }
        ]}
        fill={[
            {
                match: {
                    id: 'fries'
                },
                id: 'dots'
            },
            {
                match: {
                    id: 'sandwich'
                },
                id: 'lines'
            }
        ]}
        axisTop={null}
        axisRight={null}
        axisLeft={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'Timespent',
            legendPosition: 'middle',
            legendOffset: -40
        }}
        enableGridY={false}
        enableLabel={false}
        labelSkipWidth={10}
        labelSkipHeight={12}
        labelTextColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
        motionConfig="wobbly"
        role="application"
        ariaLabel="Nivo bar chart demo"
        barAriaLabel={function (e) { return e.id + ": " + e.formattedValue + " in Editor: " + e.indexValue; }}
    />
);
