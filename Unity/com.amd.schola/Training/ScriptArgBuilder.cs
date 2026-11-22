// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;
using System.Globalization;

namespace AMD.Schola.Training
{
    /// <summary>
    /// Builds a command line for a training script.
    /// </summary>
    public sealed class ScriptArgBuilder
    {
        readonly List<string> m_Args = new List<string>();

        public ScriptArgBuilder AddPositionalArgument(string value)
        {
            m_Args.Add(Quote(value));
            return this;
        }

        public ScriptArgBuilder AddFlag(string name)
        {
            m_Args.Add("--" + name);
            return this;
        }

        public ScriptArgBuilder AddStringArg(string name, string value)
        {
            m_Args.Add("--" + name);
            m_Args.Add(Quote(value));
            return this;
        }

        public ScriptArgBuilder AddIntArg(string name, int value)
        {
            m_Args.Add("--" + name);
            m_Args.Add(value.ToString(CultureInfo.InvariantCulture));
            return this;
        }

        public string Build()
        {
            return string.Join(" ", m_Args);
        }

        /// <summary>
        /// Quotes a value that is empty or contains spaces or quotes
        /// </summary>
        public static string Quote(string value)
        {
            if (value == null)
            {
                value = "";
            }

            if (value.Length > 0 && value.IndexOfAny(new[] { ' ', '\t', '"' }) < 0)
            {
                return value;
            }

            return "\"" + value.TrimEnd('\\').Replace("\"", "\\\"") + "\"";
        }
    }
}
