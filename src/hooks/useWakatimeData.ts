import { useState, useEffect } from "react";
import { useMemo } from "react";

interface DataItem {
    name: string;
    total_seconds: number;
}

export interface WakatimeData {
    languages: DataItem[];
    editors: DataItem[];
}

interface PieItem {
    id: string;
    label: string;
    value: number;
}

interface BarItem {
    name: string;
    value: number;
}

export function useWakatimeData () {
    const [data, setData] = useState(null);
    useEffect(() => {
        fetch('/api/wakatime.ts').then(async response => {
            setData(await response.json());
        })
    }, []);

    // Cache processed data
    const [languages, editors] = useMemo(() => {
        if (data) {
            let languages = data.languages.map(x => ({
                id: x.name,
                label: x.name,
                value: Math.round(x.total_seconds / 60)
            } as PieItem));
            languages = makeOthers(languages);

            const editors = data.editors.map(x => ({
                name: x.name,
                value: Math.round(x.total_seconds / 60)
            } as BarItem));

            console.info({ languages, editors });
            return [languages, editors];
        }
        return [[], []];
    }, [data]);

    return {languages, editors};
}

function makeOthers(data: PieItem[]) {
    const MAX_RATIO = 0.1;
    data = data.slice();
    const totalValue = data.reduce((v, x) => x.value + v, 0);
    data.sort((a, b) => a.value - b.value);
    let i = 0;
    let iSum = 0;
    for (; i < data.length; i++) {
        if (iSum + data[i].value > totalValue * MAX_RATIO) {
            break;
        } else {
            iSum += data[i].value;
        }
    }
    if (i > 0) {
        data.splice(0, i);
        data.unshift({
            id: "Others",
            label: "Others",
            value: iSum,
        } as any);
    }
    return data.sort((a, b) => b.value - a.value);
}