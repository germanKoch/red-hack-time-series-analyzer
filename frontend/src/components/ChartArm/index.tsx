//	import am5 from '@amcharts/amcharts5';
import { useLayoutEffect } from 'react'

import * as am5 from '@amcharts/amcharts5'
import am5themes_Animated from '@amcharts/amcharts5/themes/Animated'
import * as am5xy from '@amcharts/amcharts5/xy'

import { Point } from '@/api'

type ChartProps = {
	data: Point[]
}

export const Chart: React.FC<ChartProps> = ({ data }) => {
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

		const switcher = true

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
				categoryField: 'point',
				renderer: xRenderer,
				tooltip: am5.Tooltip.new(root, {}),
			}),
		)

		xAxis.data.setAll(data)

		const yAxis = chart.yAxes.push(
			am5xy.ValueAxis.new(root, {
				// maxPrecision: 0,
				renderer: am5xy.AxisRendererY.new(root, {}),
			}),
		)

		const series = chart.series.push(
			am5xy.LineSeries.new(root, {
				xAxis: xAxis,
				yAxis: yAxis,
				valueYField: 'value',
				categoryXField: 'point',
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

		// series.fills.template.setAll({
		// 	visible: true,
		// 	fillOpacity: 0.5,
		// 	templateField: 'fillSettings',
		// })

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
	}, [data])

	return <div id="chartdiv" style={{ width: '100%', height: '500px' }}></div>
}
