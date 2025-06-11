<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.4-Białowieża">
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option type="QString" name="name" value=""/>
      <Option name="properties"/>
      <Option type="QString" name="type" value="collection"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedInResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer opacity="1" type="singlebandpseudocolor" band="1" classificationMin="0" classificationMax="100">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader maximumValue="100" classificationMode="1" colorRampType="INTERPOLATED" clip="0" labelPrecision="0" minimumValue="0">
          <colorramp name="[source]" type="gradient">
            <Option type="Map">
              <Option type="QString" name="color1" value="#ffffff"/>
              <Option type="QString" name="color2" value="#ff0000"/>
              <Option type="QString" name="discrete" value="0"/>
              <Option type="QString" name="rampType" value="gradient"/>
            </Option>
          </colorramp>
          <item label="0%" alpha="0" color="#ffffff" value="0"/>
          <item label="50%" alpha="128" color="#ff0000" value="50"/>
          <item label="100%" alpha="255" color="#ff0000" value="100"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeGreen="128" invertColors="0" colorizeBlue="128" grayscaleMode="0" colorizeOn="0" saturation="0" colorizeRed="255" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
