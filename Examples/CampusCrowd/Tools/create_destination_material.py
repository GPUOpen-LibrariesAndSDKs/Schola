"""Create the instance-colored material used by the two campus demo presets."""
import unreal

path = "/Game/Campus/M_DestinationStudents"
editor = unreal.MaterialEditingLibrary
material = unreal.load_asset(path)
if material is None:
    material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        "M_DestinationStudents", "/Game/Campus", unreal.Material, unreal.MaterialFactoryNew())
editor.delete_all_material_expressions(material)
channels = []
for index in range(3):
    channel = editor.create_material_expression(material, unreal.MaterialExpressionPerInstanceCustomData)
    channel.set_editor_property("data_index", index)
    channels.append(channel)
rg = editor.create_material_expression(material, unreal.MaterialExpressionAppendVector)
editor.connect_material_expressions(channels[0], "", rg, "A")
editor.connect_material_expressions(channels[1], "", rg, "B")
rgb = editor.create_material_expression(material, unreal.MaterialExpressionAppendVector)
editor.connect_material_expressions(rg, "", rgb, "A")
editor.connect_material_expressions(channels[2], "", rgb, "B")
# VertexInterpolator carries per-instance values to pixel shader inputs.
color = editor.create_material_expression(material, unreal.MaterialExpressionVertexInterpolator)
if not editor.connect_material_expressions(rgb, "", color, "VS"):
    raise RuntimeError("Could not connect the destination colors to the vertex interpolator")
material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
editor.connect_material_property(color, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
editor.set_material_usage(material, unreal.MaterialUsage.MATUSAGE_INSTANCED_STATIC_MESHES)
editor.recompile_material(material)
assert unreal.EditorAssetLibrary.save_loaded_asset(material)
name = "M_DestinationBuildings"
buildings = unreal.load_asset("/Game/Campus/" + name)
if buildings is None:
    buildings = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, "/Game/Campus", unreal.Material, unreal.MaterialFactoryNew())
editor.delete_all_material_expressions(buildings)
buildings.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
buildings.set_editor_property("two_sided", True)
tint = editor.create_material_expression(buildings, unreal.MaterialExpressionVertexColor)
editor.connect_material_property(tint, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
editor.recompile_material(buildings)
assert unreal.EditorAssetLibrary.save_loaded_asset(buildings)
unreal.log("Campus destination materials ready")
