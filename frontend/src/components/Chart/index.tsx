//	import am5 from '@amcharts/amcharts5';
import { useLayoutEffect } from 'react'

import * as am5 from '@amcharts/amcharts5'
import am5themes_Animated from '@amcharts/amcharts5/themes/Animated'
import * as am5xy from '@amcharts/amcharts5/xy'

export const Chart = () => {
	console.log('sdgfsdg')
	//= Root.new("chartdiv");
	useLayoutEffect(() => {
		const root = am5.Root.new('chartdiv')

		// Set themes
		// https://www.amcharts.com/docs/v5/concepts/themes/
		root.setThemes([am5themes_Animated.new(root)])

		// Create chart
		// https://www.amcharts.com/docs/v5/charts/xy-chart/
		const chart = root.container.children.push(
			am5xy.XYChart.new(root, {
				panX: true,
				panY: true,
				wheelX: 'panX',
				wheelY: 'zoomX',
				layout: root.verticalLayout,
				pinchZoomX: true,
				paddingLeft: 0,
			}),
		)

		// Add cursor
		// https://www.amcharts.com/docs/v5/charts/xy-chart/cursor/
		const cursor = chart.set(
			'cursor',
			am5xy.XYCursor.new(root, {
				behavior: 'none',
			}),
		)
		cursor.lineY.set('visible', false)

		const colorSet = am5.ColorSet.new(root, {})

		// The data
		const data = [
			{
				time: '2014',
				value: 23.5,
				strokeSettings: {
					stroke: colorSet.getIndex(0),
				},
				fillSettings: {
					fill: colorSet.getIndex(0),
				},
				bulletSettings: {
					fill: colorSet.getIndex(0),
				},
			},
			{
				time: '2015',
				value: 26,
				bulletSettings: {
					fill: colorSet.getIndex(0),
				},
			},
			{
				time: '2016',
				value: 30,
				bulletSettings: {
					fill: colorSet.getIndex(0),
				},
			},
			{
				time: '2017',
				value: 20,
				bulletSettings: {
					fill: colorSet.getIndex(0),
				},
			},
			{
				time: '2018',
				value: 30,
				strokeSettings: {
					stroke: colorSet.getIndex(3),
				},
				fillSettings: {
					fill: colorSet.getIndex(3),
				},
				bulletSettings: {
					fill: colorSet.getIndex(3),
				},
			},
			{
				time: '2019',
				value: 30,
				bulletSettings: {
					fill: colorSet.getIndex(3),
				},
			},
			{
				time: '2020',
				value: 31,
				bulletSettings: {
					fill: colorSet.getIndex(3),
				},
			},
			{
				time: '2021',
				value: 34,
				strokeSettings: {
					stroke: colorSet.getIndex(6),
				},
				fillSettings: {
					fill: colorSet.getIndex(6),
				},
				bulletSettings: {
					fill: colorSet.getIndex(6),
				},
			},
			{
				time: '2022',
				value: 33,
				bulletSettings: {
					fill: colorSet.getIndex(6),
				},
			},
			{
				time: '2023',
				value: 34,
				bulletSettings: {
					fill: colorSet.getIndex(6),
				},
			},
			{
				time: '2024',
				value: 36,
				bulletSettings: {
					fill: colorSet.getIndex(7),
				},
				strokeSettings: {
					stroke: colorSet.getIndex(7),
				},
				fillSettings: {
					fill: colorSet.getIndex(7),
				},
			},
			{
				time: '2025',
				value: 33,
				bulletSettings: {
					fill: colorSet.getIndex(7),
				},
			},
			{
				time: '2026',
				value: 34,
				bulletSettings: {
					fill: colorSet.getIndex(7),
				},
			},
			{
				time: '2027',
				value: 36,
				bulletSettings: {
					fill: colorSet.getIndex(7),
				},
			},
		]

		let switcher = true

		const newData = data.map((point) => {
			if (point.isErrorEdge === true) {
				switcher = !switcher
			}

			return {
				...point,
				bulletSettings: {
					fill: switcher ? colorSet.getIndex(7) : colorSet.getIndex(1),
				},
				strokeSettings: {
					stroke: switcher ? colorSet.getIndex(7) : colorSet.getIndex(1),
				},
				fillSettings: {
					fill: switcher ? colorSet.getIndex(7) : colorSet.getIndex(1),
				},
			}
		})
		// Create axes
		// https://www.amcharts.com/docs/v5/charts/xy-chart/axes/
		const xRenderer = am5xy.AxisRendererX.new(root, {
			minorGridEnabled: true,
			minGridDistance: 80,
		})
		xRenderer.grid.template.set('location', 0.5)
		xRenderer.labels.template.setAll({
			location: 0.5,
			multiLocation: 0.5,
		})

		const xAxis = chart.xAxes.push(
			am5xy.CategoryAxis.new(root, {
				categoryField: 'time',
				renderer: xRenderer,
				tooltip: am5.Tooltip.new(root, {}),
			}),
		)

		xAxis.data.setAll(data)

		const yAxis = chart.yAxes.push(
			am5xy.ValueAxis.new(root, {
				maxPrecision: 0,
				renderer: am5xy.AxisRendererY.new(root, {}),
			}),
		)

		const series = chart.series.push(
			am5xy.LineSeries.new(root, {
				xAxis: xAxis,
				yAxis: yAxis,
				valueYField: 'value',
				categoryXField: 'time',
				tooltip: am5.Tooltip.new(root, {
					labelText: '{valueY}',
					dy: -5,
				}),
			}),
		)

		series.strokes.template.setAll({
			templateField: 'strokeSettings',
			strokeWidth: 2,
		})

		series.fills.template.setAll({
			visible: true,
			fillOpacity: 0.5,
			templateField: 'fillSettings',
		})

		series.bullets.push(function () {
			return am5.Bullet.new(root, {
				sprite: am5.Circle.new(root, {
					templateField: 'bulletSettings',
					radius: 5,
				}),
			})
		})

		series.data.setAll(data)
		series.appear(1000)

		// Add scrollbar
		// https://www.amcharts.com/docs/v5/charts/xy-chart/scrollbars/
		chart.set(
			'scrollbarX',
			am5.Scrollbar.new(root, {
				orientation: 'horizontal',
				marginBottom: 20,
			}),
		)

		// Make stuff animate on load
		// https://www.amcharts.com/docs/v5/concepts/animations/
		chart.appear(1000, 100)

		return () => {
			root.dispose()
		}
	}, [])

	return <div id="chartdiv" style={{ width: '100%', height: '500px' }}></div>
}
