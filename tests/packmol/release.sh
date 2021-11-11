#!/bin/bash
####################################################################################################

# Software name:

package=packmol

# Release version, read from command line:
version="$1"

# GIT URL:

giturl=https://github.com/m3g/packmol

# Name of file containing version number

versionfile=./title.f90

####################################################################################################

#git log --pretty=oneline 16.323...16.330 | awk '{$1=""; print "-"$0}'

year=`date +%y`
day=`date +%j`
#version="${year:0:1}${year:1:1}.$day"
if [[ $version < " " ]]; then
  echo "ERROR: Please provide version number, with: ./release.sh 20.1.1"
  exit
fi

file="$package-$version.tar.gz" 
echo "Will create file: $file"

cat $versionfile | sed -e "s/Version.*/Version\ $version \')\")/" > version_title_temp.f90
\mv -f version_title_temp.f90 $versionfile

git add -A .
git commit -m "Changed version file to $version"
git tag -a "v$version" -m "Release $version"
git push origin master tag "v$version"

today=`date +"%b %d, %Y"`
changelog="https://github.com/m3g/$package/releases/tag/v$version"
newline="<tr><td width=190px valign=top><a href=$giturl/archive/v$version.tar.gz> $file </a></td><td> Released on $today - <a target=newpage href=$changelog> [change log at github] </a></td></tr>"

echo "------------------------------"
echo "CREATING RELEASE IN HOME-PAGE:"
echo "------------------------------"
mkdir TEMP
cd TEMP
wget https://github.com/m3g/packmol/archive/v$version.tar.gz 
tar -xf v$version.tar.gz
mv packmol-$version packmol
tar -cf packmol.tar ./packmol
gzip packmol.tar
#scp packmol.tar.gz martinez@ssh.ime.unicamp.br:./public_html/packmol/
\cp -f packmol.tar.gz ~/public_html/m3g/packmol/packmol.tar.gz
cd ..
\rm -rf ./TEMP

echo "----------------------"
echo "CHANGE LOG:"
echo "----------------------"
range=`git tag | tail -n 2 | xargs | sed 's! !...!'`
git log --pretty=oneline $range | awk '{$1=""; print "-"$0}'
echo "----------------------"

echo " Done. " 

