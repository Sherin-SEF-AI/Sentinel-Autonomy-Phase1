"""HD map parsers for OpenDRIVE and Lanelet2 formats."""

import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.data_structures import Lane, MapFeature

logger = logging.getLogger(__name__)


class HDMapParser(ABC):
    """Abstract base class for HD map parsers."""
    
    @abstractmethod
    def parse(self, map_file: str) -> Dict:
        """Parse HD map file and return map data structure.
        
        Args:
            map_file: Path to map file
            
        Returns:
            Dictionary containing:
                - lanes: Dict[str, Lane]
                - features: List[MapFeature]
                - metadata: Dict with map info
        """
        pass


class OpenDRIVEParser(HDMapParser):
    """Parser for OpenDRIVE format HD maps."""
    
    def __init__(self):
        """Initialize OpenDRIVE parser."""
        self.logger = logging.getLogger(__name__ + '.OpenDRIVEParser')
    
    def parse(self, map_file: str) -> Dict:
        """Parse OpenDRIVE XML file.
        
        Args:
            map_file: Path to .xodr file
            
        Returns:
            Dictionary with lanes, features, and metadata
        """
        self.logger.info(f"Parsing OpenDRIVE map: {map_file}")
        
        try:
            tree = ET.parse(map_file)
            root = tree.getroot()
            
            # Extract metadata
            header = root.find('header')
            metadata = self._parse_header(header) if header is not None else {}
            
            # Parse roads and lanes
            lanes = {}
            features = []
            
            for road in root.findall('road'):
                road_lanes, road_features = self._parse_road(road)
                lanes.update(road_lanes)
                features.extend(road_features)
            
            self.logger.info(f"Parsed {len(lanes)} lanes and {len(features)} features")
            
            return {
                'lanes': lanes,
                'features': features,
                'metadata': metadata,
                'format': 'opendrive'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse OpenDRIVE map: {e}")
            raise
    
    def _parse_header(self, header: ET.Element) -> Dict:
        """Parse OpenDRIVE header."""
        return {
            'name': header.get('name', 'Unknown'),
            'version': header.get('revMajor', '1') + '.' + header.get('revMinor', '0'),
            'date': header.get('date', ''),
            'north': float(header.get('north', 0)),
            'south': float(header.get('south', 0)),
            'east': float(header.get('east', 0)),
            'west': float(header.get('west', 0)),
        }
    
    def _parse_road(self, road: ET.Element) -> Tuple[Dict[str, Lane], List[MapFeature]]:
        """Parse a road element.
        
        Returns:
            Tuple of (lanes dict, features list)
        """
        road_id = road.get('id', 'unknown')
        road_length = float(road.get('length', 0))
        
        lanes = {}
        features = []
        
        # Parse reference line (road geometry)
        plan_view = road.find('planView')
        if plan_view is not None:
            reference_line = self._parse_plan_view(plan_view, road_length)
        else:
            reference_line = []
        
        # Parse lanes
        lanes_section = road.find('lanes')
        if lanes_section is not None:
            for lane_section in lanes_section.findall('laneSection'):
                section_lanes = self._parse_lane_section(
                    lane_section, road_id, reference_line
                )
                lanes.update(section_lanes)
        
        # Parse objects (signs, lights, etc.)
        objects = road.find('objects')
        if objects is not None:
            for obj in objects.findall('object'):
                feature = self._parse_object(obj, road_id, reference_line)
                if feature:
                    features.append(feature)
        
        # Parse signals (traffic lights, signs)
        signals = road.find('signals')
        if signals is not None:
            for signal in signals.findall('signal'):
                feature = self._parse_signal(signal, road_id, reference_line)
                if feature:
                    features.append(feature)
        
        return lanes, features
    
    def _parse_plan_view(self, plan_view: ET.Element, road_length: float) -> List[Tuple[float, float, float]]:
        """Parse road reference line geometry.
        
        Returns:
            List of (x, y, heading) points along reference line
        """
        points = []
        
        for geometry in plan_view.findall('geometry'):
            s = float(geometry.get('s', 0))
            x = float(geometry.get('x', 0))
            y = float(geometry.get('y', 0))
            hdg = float(geometry.get('hdg', 0))
            length = float(geometry.get('length', 0))
            
            # Sample points along geometry
            # Simplified: just use start point
            # In production, would interpolate based on geometry type (line, arc, spiral, etc.)
            points.append((x, y, hdg))
        
        return points
    
    def _parse_lane_section(self, lane_section: ET.Element, road_id: str, 
                           reference_line: List[Tuple[float, float, float]]) -> Dict[str, Lane]:
        """Parse a lane section."""
        lanes = {}
        s_start = float(lane_section.get('s', 0))
        
        # Parse center, left, and right lanes
        for side in ['center', 'left', 'right']:
            side_elem = lane_section.find(side)
            if side_elem is None:
                continue
            
            for lane_elem in side_elem.findall('lane'):
                lane_id_num = lane_elem.get('id', '0')
                lane_id = f"{road_id}_{lane_id_num}_{s_start}"
                lane_type = lane_elem.get('type', 'driving')
                
                # Skip center lane (reference)
                if lane_id_num == '0':
                    continue
                
                # Get lane width
                width_elem = lane_elem.find('width')
                width = float(width_elem.get('a', 3.5)) if width_elem is not None else 3.5
                
                # Get speed limit
                speed_elem = lane_elem.find('speed')
                speed_limit = float(speed_elem.get('max', 0)) if speed_elem is not None else None
                if speed_limit == 0:
                    speed_limit = None
                
                # Generate lane geometry (simplified)
                centerline, left_boundary, right_boundary = self._generate_lane_geometry(
                    reference_line, int(lane_id_num), width
                )
                
                # Get predecessors and successors
                link = lane_elem.find('link')
                predecessors = []
                successors = []
                if link is not None:
                    pred = link.find('predecessor')
                    if pred is not None:
                        predecessors.append(f"{road_id}_{pred.get('id', '0')}")
                    succ = link.find('successor')
                    if succ is not None:
                        successors.append(f"{road_id}_{succ.get('id', '0')}")
                
                lane = Lane(
                    lane_id=lane_id,
                    centerline=centerline,
                    left_boundary=left_boundary,
                    right_boundary=right_boundary,
                    width=width,
                    speed_limit=speed_limit,
                    lane_type=lane_type,
                    predecessors=predecessors,
                    successors=successors
                )
                
                lanes[lane_id] = lane
        
        return lanes
    
    def _generate_lane_geometry(self, reference_line: List[Tuple[float, float, float]], 
                                lane_id: int, width: float) -> Tuple[List, List, List]:
        """Generate lane centerline and boundaries from reference line.
        
        Args:
            reference_line: Reference line points (x, y, heading)
            lane_id: Lane ID (positive = left, negative = right)
            width: Lane width
            
        Returns:
            Tuple of (centerline, left_boundary, right_boundary)
        """
        if not reference_line:
            return [], [], []
        
        centerline = []
        left_boundary = []
        right_boundary = []
        
        # Lateral offset from reference line
        lateral_offset = lane_id * width
        
        for x, y, hdg in reference_line:
            # Calculate perpendicular offset
            offset_x = -lateral_offset * np.sin(hdg)
            offset_y = lateral_offset * np.cos(hdg)
            
            center_x = x + offset_x
            center_y = y + offset_y
            
            # Calculate boundaries
            left_offset_x = -(lateral_offset + width/2) * np.sin(hdg)
            left_offset_y = (lateral_offset + width/2) * np.cos(hdg)
            
            right_offset_x = -(lateral_offset - width/2) * np.sin(hdg)
            right_offset_y = (lateral_offset - width/2) * np.cos(hdg)
            
            centerline.append((center_x, center_y, 0.0))
            left_boundary.append((x + left_offset_x, y + left_offset_y, 0.0))
            right_boundary.append((x + right_offset_x, y + right_offset_y, 0.0))
        
        return centerline, left_boundary, right_boundary
    
    def _parse_object(self, obj: ET.Element, road_id: str, 
                     reference_line: List[Tuple[float, float, float]]) -> Optional[MapFeature]:
        """Parse a road object (crosswalk, barrier, etc.)."""
        obj_id = obj.get('id', 'unknown')
        obj_type = obj.get('type', 'unknown')
        s = float(obj.get('s', 0))
        t = float(obj.get('t', 0))
        
        # Get position from reference line
        if reference_line:
            x, y, hdg = reference_line[0]  # Simplified
            # Apply lateral offset
            pos_x = x - t * np.sin(hdg)
            pos_y = y + t * np.cos(hdg)
        else:
            pos_x, pos_y = 0, 0
        
        # Map object types to feature types
        feature_type_map = {
            'crosswalk': 'crosswalk',
            'barrier': 'boundary',
            'pole': 'pole',
        }
        
        feature_type = feature_type_map.get(obj_type, 'object')
        
        return MapFeature(
            feature_id=f"{road_id}_{obj_id}",
            type=feature_type,
            position=(pos_x, pos_y, 0.0),
            attributes={'object_type': obj_type},
            geometry=[(pos_x, pos_y)]
        )
    
    def _parse_signal(self, signal: ET.Element, road_id: str,
                     reference_line: List[Tuple[float, float, float]]) -> Optional[MapFeature]:
        """Parse a traffic signal (sign or light)."""
        signal_id = signal.get('id', 'unknown')
        signal_type = signal.get('type', 'unknown')
        subtype = signal.get('subtype', '')
        s = float(signal.get('s', 0))
        t = float(signal.get('t', 0))
        
        # Get position
        if reference_line:
            x, y, hdg = reference_line[0]  # Simplified
            pos_x = x - t * np.sin(hdg)
            pos_y = y + t * np.cos(hdg)
        else:
            pos_x, pos_y = 0, 0
        
        # Determine feature type
        if 'light' in signal_type.lower() or signal_type.startswith('1000'):
            feature_type = 'light'
        else:
            feature_type = 'sign'
        
        return MapFeature(
            feature_id=f"{road_id}_{signal_id}",
            type=feature_type,
            position=(pos_x, pos_y, 0.0),
            attributes={
                'signal_type': signal_type,
                'subtype': subtype,
            },
            geometry=[(pos_x, pos_y)]
        )


class Lanelet2Parser(HDMapParser):
    """Parser for Lanelet2 format HD maps."""
    
    def __init__(self):
        """Initialize Lanelet2 parser."""
        self.logger = logging.getLogger(__name__ + '.Lanelet2Parser')
    
    def parse(self, map_file: str) -> Dict:
        """Parse Lanelet2 OSM XML file.
        
        Args:
            map_file: Path to .osm file
            
        Returns:
            Dictionary with lanes, features, and metadata
        """
        self.logger.info(f"Parsing Lanelet2 map: {map_file}")
        
        try:
            tree = ET.parse(map_file)
            root = tree.getroot()
            
            # Parse nodes (points)
            nodes = self._parse_nodes(root)
            
            # Parse ways (polylines)
            ways = self._parse_ways(root, nodes)
            
            # Parse relations (lanelets)
            lanes, features = self._parse_relations(root, ways)
            
            metadata = {
                'format': 'lanelet2',
                'num_nodes': len(nodes),
                'num_ways': len(ways),
            }
            
            self.logger.info(f"Parsed {len(lanes)} lanes and {len(features)} features")
            
            return {
                'lanes': lanes,
                'features': features,
                'metadata': metadata,
                'format': 'lanelet2'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse Lanelet2 map: {e}")
            raise
    
    def _parse_nodes(self, root: ET.Element) -> Dict[str, Tuple[float, float]]:
        """Parse OSM nodes (points)."""
        nodes = {}
        for node in root.findall('node'):
            node_id = node.get('id')
            lat = float(node.get('lat', 0))
            lon = float(node.get('lon', 0))
            # Convert lat/lon to local coordinates (simplified - would use proper projection)
            x = lon * 111320  # Approximate meters
            y = lat * 110540
            nodes[node_id] = (x, y)
        return nodes
    
    def _parse_ways(self, root: ET.Element, nodes: Dict) -> Dict[str, List[Tuple[float, float]]]:
        """Parse OSM ways (polylines)."""
        ways = {}
        for way in root.findall('way'):
            way_id = way.get('id')
            points = []
            for nd in way.findall('nd'):
                ref = nd.get('ref')
                if ref in nodes:
                    points.append(nodes[ref])
            ways[way_id] = points
        return ways
    
    def _parse_relations(self, root: ET.Element, ways: Dict) -> Tuple[Dict[str, Lane], List[MapFeature]]:
        """Parse OSM relations (lanelets and regulatory elements)."""
        lanes = {}
        features = []
        
        for relation in root.findall('relation'):
            rel_type = None
            for tag in relation.findall('tag'):
                if tag.get('k') == 'type':
                    rel_type = tag.get('v')
                    break
            
            if rel_type == 'lanelet':
                lane = self._parse_lanelet(relation, ways)
                if lane:
                    lanes[lane.lane_id] = lane
            elif rel_type == 'regulatory_element':
                feature = self._parse_regulatory_element(relation, ways)
                if feature:
                    features.append(feature)
        
        return lanes, features
    
    def _parse_lanelet(self, relation: ET.Element, ways: Dict) -> Optional[Lane]:
        """Parse a lanelet relation."""
        lanelet_id = relation.get('id')
        
        # Get left and right boundaries
        left_way_id = None
        right_way_id = None
        
        for member in relation.findall('member'):
            role = member.get('role')
            ref = member.get('ref')
            if role == 'left':
                left_way_id = ref
            elif role == 'right':
                right_way_id = ref
        
        if not left_way_id or not right_way_id:
            return None
        
        left_boundary = ways.get(left_way_id, [])
        right_boundary = ways.get(right_way_id, [])
        
        if not left_boundary or not right_boundary:
            return None
        
        # Calculate centerline
        min_len = min(len(left_boundary), len(right_boundary))
        centerline = []
        for i in range(min_len):
            lx, ly = left_boundary[i]
            rx, ry = right_boundary[i]
            cx = (lx + rx) / 2
            cy = (ly + ry) / 2
            centerline.append((cx, cy, 0.0))
        
        # Add z coordinate to boundaries
        left_boundary_3d = [(x, y, 0.0) for x, y in left_boundary]
        right_boundary_3d = [(x, y, 0.0) for x, y in right_boundary]
        
        # Calculate width
        if centerline:
            lx, ly = left_boundary[0]
            rx, ry = right_boundary[0]
            width = np.sqrt((lx - rx)**2 + (ly - ry)**2)
        else:
            width = 3.5
        
        # Parse tags
        lane_type = 'driving'
        speed_limit = None
        for tag in relation.findall('tag'):
            k = tag.get('k')
            v = tag.get('v')
            if k == 'subtype':
                lane_type = v
            elif k == 'speed_limit':
                try:
                    speed_limit = float(v)
                except:
                    pass
        
        return Lane(
            lane_id=lanelet_id,
            centerline=centerline,
            left_boundary=left_boundary_3d,
            right_boundary=right_boundary_3d,
            width=width,
            speed_limit=speed_limit,
            lane_type=lane_type,
            predecessors=[],  # Would parse from regulatory elements
            successors=[]
        )
    
    def _parse_regulatory_element(self, relation: ET.Element, ways: Dict) -> Optional[MapFeature]:
        """Parse a regulatory element (sign, light, etc.)."""
        elem_id = relation.get('id')
        
        # Parse tags to determine type
        elem_type = 'unknown'
        attributes = {}
        for tag in relation.findall('tag'):
            k = tag.get('k')
            v = tag.get('v')
            attributes[k] = v
            if k == 'subtype':
                elem_type = v
        
        # Get position from first member
        position = (0.0, 0.0, 0.0)
        geometry = []
        for member in relation.findall('member'):
            ref = member.get('ref')
            if ref in ways and ways[ref]:
                x, y = ways[ref][0]
                position = (x, y, 0.0)
                geometry = ways[ref]
                break
        
        # Map to feature type
        if 'traffic_light' in elem_type:
            feature_type = 'light'
        elif 'traffic_sign' in elem_type:
            feature_type = 'sign'
        else:
            feature_type = 'regulatory'
        
        return MapFeature(
            feature_id=elem_id,
            type=feature_type,
            position=position,
            attributes=attributes,
            geometry=geometry
        )
