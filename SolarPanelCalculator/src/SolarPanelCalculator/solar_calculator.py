import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Set


class SolarPanelCalculator:
    """
    SolarPanelCalculator
    --------------------
    Responsible for:
      - Computing rafter positions (vertical lines)
      - Computing mount coordinates (points where panels are attached to rafters)
      - Computing joint coordinates (connectors between adjacent panels)
      - Plotting the layout

    The calculator enforces structural constraints:
      - Edge clearance: minimum 2 units from panel edges
      - Cantilever limit: maximum 16 units unsupported overhang
      - Span limit: maximum 48 units between consecutive supports
      - Shared connections: joints can connect up to 4 panels in grid layouts
    """

    def __init__(
        self,
        rafter_x0: float,
        rafter_spacing: float = 16.0,
        panel_width: float = 44.7,
        panel_height: float = 71.1,
        edge_clearance: float = 2.0,
        centeliver_limit: float = 16.0,
        span_limit: float = 48.0
    ):
        """
        Initialize calculator

        Parameters:
            rafter_x0 (float):
                X coordinate of the first rafter.
            rafter_spacing (float, default=16.0):
                Distance between consecutive rafters.
            panel_width (float, default=44.7):
                Panel width (units).
            panel_height (float, default=71.1):
                Panel height (units).
            edge_clearance (float, default=2.0):
                Minimum horizontal distance between a mount and panel left/right edge.
            centeliver_limit (float, default=16.0):
                Max allowed distance from panel edge to the nearest mount (cantilever limit).
            span_limit (float, default=48.0):
                Maximum horizontal distance between two consecutive supports on a panel.
        """
        # Validate configuration
        if rafter_spacing <= 0:
            raise ValueError("Rafter spacing must be positive")
        if rafter_x0 < 0:
            raise ValueError("Rafter x0 must be non-negative")

        self.rafter_x0 = rafter_x0
        self.rafter_spacing = rafter_spacing
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.edge_clearance = edge_clearance
        self.centeliver_limit = centeliver_limit
        self.span_limit = span_limit

        # These will be populated by calculate()
        self.rafter_coordinates: List[float] = []
        self.mount_coordinates: List[Dict[str, float]] = []
        self.joint_coordinates: List[Dict[str, float]] = []

    # ---------------------
    # Public API
    # ---------------------
    def get_coordinates(self, panel_left_top_coordinates: List[Dict[str, float]]) -> Dict:
        """
        Calculate rafters, mounts and joints and return the results.

        Input:
            panel_left_top_coordinates: list of dicts with keys {"x": float, "y": float}
                Coordinates represent top-left corner of each panel.

        Returns:
            Dictionary with keys:
              - 'mounts': List of mount coordinates {'x': float, 'y': float}
              - 'joints': List of joint coordinates {'x': float, 'y': float}

        Raises:
            TypeError: If panel_left_top_coordinates is not a list of dicts.
            ValueError: If panel data is invalid.
        """
        # Validate input
        self._validate_input(panel_left_top_coordinates)

        # Calculate all elements
        self._calculate_rafters(panel_left_top_coordinates)
        self._calculate_mounts(panel_left_top_coordinates)
        self._calculate_joints(panel_left_top_coordinates)

        # Return results
        result = {
            'mount_coordinates': self.mount_coordinates,
            'joint_coordinates': self.joint_coordinates
        }
        return result

    def plot_layout(self, panel_left_top_coordinates: List[Dict[str, float]]) -> None:
        """
        Calculate rafters, mounts, joints and plot the layout.

        Input:
            panel_left_top_coordinates: list of dicts with keys {"x": float, "y": float}
        """
        # Validate input
        self._validate_input(panel_left_top_coordinates)

        # Calculate all structural elements
        self._calculate_rafters(panel_left_top_coordinates)
        self._calculate_mounts(panel_left_top_coordinates)
        self._calculate_joints(panel_left_top_coordinates)

        # Plot the layout
        self._plot_output(panel_left_top_coordinates)

    # ---------------------
    # Input Validation
    # ---------------------
    def _validate_input(self, panel_left_top_coordinates: List[Dict[str, float]]) -> None:
        """
        Validate input panel coordinates.

        Parameters:
            panel_left_top_coordinates: list of panel position dicts

        Raises:
            TypeError: If input is not a list of dicts
            ValueError: If panel data is invalid
        """
        if not isinstance(panel_left_top_coordinates, list):
            raise TypeError(
                f"panel_left_top_coordinates must be a list, got {type(panel_left_top_coordinates).__name__}"
            )

        if len(panel_left_top_coordinates) == 0:
            raise ValueError("panel_left_top_coordinates list cannot be empty")

        for idx, panel in enumerate(panel_left_top_coordinates):
            if not isinstance(panel, dict):
                raise TypeError(
                    f"Panel at index {idx}: expected dict, got {type(panel).__name__}"
                )

            if "x" not in panel or "y" not in panel:
                raise ValueError(
                    f"Panel at index {idx}: missing 'x' or 'y' coordinate"
                )

            try:
                x = float(panel["x"])
                y = float(panel["y"])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Panel at index {idx}: coordinates must be numeric. Error: {e}"
                )

            if x < 0 or y < 0:
                raise ValueError(
                    f"Panel at index {idx}: coordinates must be non-negative"
                )

    # ---------------------
    # Internal helpers
    # ---------------------
    def _calculate_rafters(self, panel_top_left_coordinates: List[Dict[str, float]]) -> None:
        """
        Compute rafter positions that cover the full horizontal extent of panels.

        Algorithm:
          - Find min left x and max right x (right = panel.x + panel_width)
          - Start from rafter_x0 and step by rafter_spacing to fill the range [x_min, x_max]
        """
        if not panel_top_left_coordinates:
            self.rafter_coordinates = []
            return

        x_values = [p["x"] for p in panel_top_left_coordinates]
        x_min = min(x_values)
        x_max = max(x_values) + self.panel_width

        current_x = self.rafter_x0
        rafter_coordinates: List[float] = []

        # Generate rafters from rafter_x0 forward
        while current_x <= x_max:
            if current_x >= x_min:
                rafter_coordinates.append(current_x)
            current_x += self.rafter_spacing

        self.rafter_coordinates = rafter_coordinates

    def _calculate_mounts(self, panel_top_left_coordinates: List[Dict[str, float]]) -> None:
        """
        Compute mounts:
          - Group panels by their y (rows)
          - Within each row create contiguous segments (merge_gap = 1.0)
          - For each segment choose minimum set of rafters that satisfy edge clearance,
            cantilever limit and span limit
          - Create two mount points per chosen rafter (bottom and top of panel)
          - Combine close vertical mounts (same rafter x, y distance < 1.0) into single mount
            located at the average y of the pair

        Result:
          - self.mount_coordinates: list of {'x': float, 'y': float} sorted by x,y
        """
        rows: Dict[float, List[Dict[str, float]]] = {}
        merge_gap = 1.0

        # Group panels by row y
        for p in panel_top_left_coordinates:
            y = p["y"]
            left = p["x"]
            right = p["x"] + self.panel_width
            rows.setdefault(y, []).append({"left": left, "right": right, "y": y})

        # Sort each row by x
        for y in rows:
            rows[y].sort(key=lambda q: q["left"])

        all_mounts: List[Dict[str, float]] = []

        # For each row build segments and select rafters
        for y, panels_row in rows.items():
            segments: List[List[Dict[str, float]]] = []
            if panels_row:
                cur_seg = [panels_row[0]]
                cur_right = panels_row[0]["right"]
                for p in panels_row[1:]:
                    gap = p["left"] - cur_right
                    if gap < merge_gap:
                        cur_seg.append(p)
                        cur_right = max(cur_right, p["right"])
                    else:
                        segments.append(cur_seg)
                        cur_seg = [p]
                        cur_right = p["right"]
                segments.append(cur_seg)

            for seg in segments:
                seg_left = min(p["left"] for p in seg)
                seg_right = max(p["right"] for p in seg)

                # Candidates are rafters that are at least edge_clearance away from panel left/right
                candidates: List[float] = []
                for r in self.rafter_coordinates:
                    for p in seg:
                        if (r >= p["left"] + self.edge_clearance) and (r <= p["right"] - self.edge_clearance):
                            candidates.append(r)
                            break
                candidates = sorted(set(candidates))

                # Choose supports
                left_bound = seg_left + self.centeliver_limit
                possible_first = [r for r in candidates if r <= left_bound]
                first = max(possible_first)
                picks: List[float] = [first]

                if seg_right - picks[-1] > self.centeliver_limit:
                    fail = False
                    while seg_right - picks[-1] > self.centeliver_limit:
                        current = picks[-1]
                        reach = current + self.span_limit
                        possibles = [r for r in candidates if r > current and r <= reach]
                        if not possibles:
                            fail = True
                            break
                        nxt = max(possibles)
                        picks.append(nxt)
                    if fail:
                        continue

                # Append bottom/top mount for each pick
                for x in picks:
                    all_mounts.append({"x": x, "y_bottom": y, "y_top": y + self.panel_height})

        # Convert to individual mount points (bottom and top)
        single_mounts: List[Dict[str, float]] = []
        for m in all_mounts:
            single_mounts.append({"x": m["x"], "y": m["y_bottom"]})
            single_mounts.append({"x": m["x"], "y": m["y_top"]})

        # Group by x to merge close vertical points
        groups: Dict[float, List[float]] = {}
        for p in single_mounts:
            x = p["x"]
            groups.setdefault(x, []).append(p["y"])

        mounts: List[Dict[str, float]] = []
        for x, y_list in groups.items():
            y_list_sorted = sorted(y_list)
            i = 0
            n = len(y_list_sorted)
            while i < n:
                if i + 1 < n and abs(y_list_sorted[i + 1] - y_list_sorted[i]) < 1.0:
                    y_avg = (y_list_sorted[i] + y_list_sorted[i + 1]) / 2.0
                    mounts.append({'x': x, 'y': y_avg})
                    i += 2
                else:
                    mounts.append({'x': x, 'y': y_list_sorted[i]})
                    i += 1

        # Sort and store
        mounts.sort(key=lambda p: (p['x'], p['y']))
        self.mount_coordinates = mounts

    def _calculate_joints(self, panel_top_left_coordinates: List[Dict[str, float]]) -> None:
        """
        Compute joints:
          - Sort panels by y then x
          - If two panels in same row are adjacent (gap in x in [0, gap_tolerance)), create seam_x midpoint
            and add top and bottom joint (y and y+panel_height)
          - Handle shared connections: if rows are vertically adjacent (gap < vertical_gap_threshold),
            merge vertical joints into single shared joint connecting up to 4 panels
          - For each seam_x merge close vertical joints (vertical gap < gap_tolerance) into single joint
            at averaged y
        """
        # Sort panels: (rounded y, rounded x)
        panels_sorted = sorted(
            panel_top_left_coordinates,
            key=lambda p: (round(p['y'], 2), round(p['x'], 2))
        )

        joints: Set[Tuple[float, float]] = set()
        gap_tolerance = 1.0

        def near(a: float, b: float, tol: float = 1e-6) -> bool:
            """Check if two floating-point values are approximately equal."""
            return abs(a - b) <= tol

        n = len(panels_sorted)
        for i in range(n):
            p_i = panels_sorted[i]
            x_i = p_i['x']
            y_i = p_i['y']

            for j in range(i + 1, n):
                p_j = panels_sorted[j]
                x_j = p_j['x']
                y_j = p_j['y']

                # If rows are far apart vertically (> gap_tolerance) break out
                if (y_j - y_i) >= gap_tolerance and (y_j - y_i) > 1e-6:
                    break

                if abs(y_j - y_i) < gap_tolerance:
                    # Ensure p_j is to the right of p_i
                    if x_j >= x_i:
                        gap = x_j - (x_i + self.panel_width)
                        # If gap in [0, gap_tolerance) -> adjacency
                        if 0 <= gap < gap_tolerance:
                            seam_x = (x_i + self.panel_width + x_j) / 2.0
                            top_joint = (round(seam_x, 2), round(y_i, 2))
                            bottom_joint = (round(seam_x, 2), round(y_i + self.panel_height, 2))
                            joints.add(top_joint)
                            joints.add(bottom_joint)

        # Map seam_x to set of y values
        seam_map: Dict[float, Set[float]] = {}
        for (sx, sy) in joints:
            seam_map.setdefault(sx, set()).add(sy)

        # Merge close vertical joints for each seam_x (shared connections support)
        for sx, ys in seam_map.items():
            ys_list = sorted(ys)
            i = 0
            while i < len(ys_list) - 1:
                y1 = ys_list[i]
                y2 = ys_list[i + 1]
                if (y2 - y1) < gap_tolerance + 1e-9:
                    # Merge vertically adjacent joints (shared joint connecting up to 4 panels)
                    merged_y = round((y1 + y2) / 2.0, 2)
                    merged = (sx, merged_y)
                    joints.discard((sx, y1))
                    joints.discard((sx, y2))
                    joints.add(merged)
                    i += 2
                else:
                    i += 1

        # Convert to list of dicts and sort
        joint_coordinates = sorted(joints, key=lambda p: (p[0], p[1]))
        self.joint_coordinates = [{'x': coord[0], 'y': coord[1]} for coord in joint_coordinates]

    def _plot_output(self, panel_top_left_coordinates: List[Dict[str, float]]) -> None:
        """
        Plot rafters (vertical lines), panels (blue rectangles), joints (grey small squares)
        and mounts (red points).
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Draw rafters
        for x in self.rafter_coordinates:
            ax.axvline(x=x, color='black', linewidth=2, alpha=0.3)

        # Draw panels as blue rectangls
        for coord in panel_top_left_coordinates:
            rect = patches.Rectangle(
                (coord["x"], coord["y"]),
                self.panel_width,
                self.panel_height,
                linewidth=1,
                edgecolor='blue',
                facecolor='blue',
                alpha=0.3
            )
            ax.add_patch(rect)

        # Draw joints as grey squares
        for coord in self.joint_coordinates:
            sq = patches.Rectangle(
                (coord['x'] - 1.5, coord['y'] - 1.5),
                3.0,
                3.0,
                linewidth=1,
                edgecolor='gray',
                facecolor='lightgray',
                alpha=1.0
            )
            ax.add_patch(sq)

        # Draw mounts as red circles
        for coord in self.mount_coordinates:
            ax.scatter(coord['x'], coord['y'], color='red', marker='o', s=100, zorder=5)

        # Plot settings
        ax.set_xlim(-10, 200)
        ax.set_ylim(-10, 250)
        ax.set_title('Solar Panel Calculator', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate (units)', fontsize=12)
        ax.set_ylabel('Y Coordinate (units)', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
