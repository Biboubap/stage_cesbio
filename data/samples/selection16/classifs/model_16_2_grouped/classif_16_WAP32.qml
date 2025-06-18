<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis hasScaleBasedVisibilityFlag="0" maxScale="0" version="3.22.4-Białowieża" minScale="1e+08" styleCategories="AllStyleCategories">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal enabled="0" fetchMode="0" mode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option type="bool" value="false" name="WMSBackgroundLayer"/>
      <Option type="bool" value="false" name="WMSPublishDataSourceUrl"/>
      <Option type="int" value="0" name="embeddedWidgets/count"/>
      <Option type="QString" value="Value" name="identify/format"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option type="QString" value="" name="name"/>
      <Option name="properties"/>
      <Option type="QString" value="collection" name="type"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling zoomedOutResamplingMethod="nearestNeighbour" enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer alphaBand="-1" nodataColor="" type="paletted" opacity="1" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry color="#000000" alpha="0" value="0" label="0"/>
        <paletteEntry color="#969696" alpha="255" value="1" label="1"/>
        <paletteEntry color="#64b464" alpha="255" value="2" label="2"/>
        <paletteEntry color="#c8c8c8" alpha="255" value="3" label="3"/>
        <paletteEntry color="#4169e1" alpha="255" value="4" label="4"/>
        <paletteEntry color="#000000" alpha="0" value="255" label="255"/>
      </colorPalette>
      <colorramp type="randomcolors" name="[source]">
        <Option/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast contrast="0" brightness="0" gamma="1"/>
    <huesaturation invertColors="0" saturation="0" colorizeRed="255" colorizeOn="0" colorizeGreen="128" colorizeBlue="128" grayscaleMode="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
