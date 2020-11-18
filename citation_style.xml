<?xml version="1.0" encoding="utf-8"?>
<style xmlns="http://purl.org/net/xbiblio/csl" class="in-text" version="1.0" demote-non-dropping-particle="sort-only" default-locale="en-US">
  <info>
    <title>Laser &amp; Photonics Reviews - Benjamin Maclellan</title>
    <id>https://csl.mendeley.com/styles/456765651/laser-and-photonics-reviews-lpr-bm</id>
    <link href="http://www.zotero.org/styles/laser-and-photonics-reviews" rel="self"/>
    <link href="http://www.zotero.org/styles/springer-physics-brackets" rel="template"/>
    <link href="http://onlinelibrary.wiley.com/journal/10.1002/(ISSN)1863-8899/homepage/2414_Author_Guidelines.html" rel="documentation"/>
    <contributor>
      <name>Benjamin Maclellan</name>
      <uri>https://www.mendeley.com/profiles/benjamin-maclellan/</uri>
    </contributor>
    <category citation-format="numeric"/>
    <category field="physics"/>
    <issn>1863-8880</issn>
    <eissn>1863-8899</eissn>
    <updated>2020-11-18T22:09:11+00:00</updated>
    <rights license="http://creativecommons.org/licenses/by-sa/3.0/">This work is licensed under a Creative Commons Attribution-ShareAlike 3.0 License</rights>
  </info>
  <macro name="author">
    <names variable="author">
      <name delimiter=", " initialize-with=". " and="text"/>
      <label form="short" prefix=" (" suffix=")"/>
      <substitute>
        <names variable="editor"/>
        <names variable="translator"/>
      </substitute>
    </names>
  </macro>
  <macro name="editor">
    <names variable="editor">
      <name delimiter=", " initialize-with=". " and="text"/>
      <label form="short" prefix=" (" suffix=")"/>
    </names>
  </macro>
  <macro name="year-date">
    <choose>
      <if variable="issued">
        <date variable="issued">
          <date-part name="year"/>
        </date>
      </if>
      <else>
        <text term="no date" form="short"/>
      </else>
    </choose>
  </macro>
  <macro name="day-date">
    <choose>
      <if variable="issued">
        <date variable="issued">
          <date-part name="day" suffix=" "/>
          <date-part name="month" form="long" suffix=" "/>
          <date-part name="year"/>
        </date>
      </if>
      <else>
        <text term="no date" form="short"/>
      </else>
    </choose>
  </macro>
  <macro name="publisher">
    <group prefix="(" suffix=")" delimiter=", ">
      <text variable="publisher"/>
      <text variable="publisher-place" text-case="title"/>
      <text macro="year-date"/>
    </group>
  </macro>
  <macro name="edition">
    <choose>
      <if is-numeric="edition">
        <group delimiter=" ">
          <number variable="edition" form="ordinal"/>
          <text term="edition" form="short"/>
        </group>
      </if>
      <else>
        <text variable="edition"/>
      </else>
    </choose>
  </macro>
  <citation collapse="citation-number">
    <sort>
      <key variable="citation-number"/>
    </sort>
    <layout vertical-align="sup" delimiter=",">
      <text variable="citation-number" vertical-align="sup"/>
    </layout>
  </citation>
  <bibliography delimiter-precedes-last="always" entry-spacing="0">
    <layout suffix=".">
      <text variable="citation-number" prefix="[" suffix="] "/>
      <text macro="author" suffix=", "/>
      <choose>
        <if type="bill book graphic legal_case legislation motion_picture report song" match="any">
          <group delimiter=" ">
            <group delimiter=", ">
              <text variable="title" text-case="title"/>
              <text macro="edition"/>
            </group>
            <group delimiter=", ">
              <text macro="publisher"/>
              <group delimiter=" ">
                <label variable="page" form="short"/>
                <text variable="page"/>
              </group>
            </group>
          </group>
        </if>
        <else-if type="chapter paper-conference" match="any">
          <group delimiter=": ">
            <text term="in"/>
            <group delimiter=", ">
              <text variable="container-title" form="short" text-case="title"/>
              <text macro="editor"/>
              <text macro="edition"/>
            </group>
            <group delimiter=", ">
              <text macro="publisher"/>
              <group delimiter=" ">
                <label variable="page" form="short"/>
                <text variable="page"/>
              </group>
            </group>
          </group>
        </else-if>
        <else-if type="patent">
          <group delimiter=" ">
            <text variable="number"/>
            <text macro="day-date" prefix="(" suffix=")"/>
          </group>
        </else-if>
        <else-if type="thesis">
          <group delimiter=", ">
            <text variable="title" text-case="title"/>
            <text variable="genre"/>
            <text variable="publisher"/>
            <text macro="year-date"/>
          </group>
        </else-if>
        <else>
          <group delimiter=" ">
            <text variable="container-title" form="short" text-case="title" font-style="italic"/>
            <text macro="year-date" font-weight="bold" suffix=","/>
            <group delimiter=" "/>
            <group delimiter=", ">
              <text variable="volume" font-style="italic" font-weight="normal" suffix=","/>
              <text variable="page" form="short"/>
            </group>
          </group>
        </else>
      </choose>
    </layout>
  </bibliography>
</style>